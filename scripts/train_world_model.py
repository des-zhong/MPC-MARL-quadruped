"""Train an offline ensemble from one dataset or a collector output root."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare_dataset(source, output, seed):
    """Merge complete episodes, deduplicate, and split without touching sources."""
    import numpy as np
    from quadruped.world_model.dataset import Episode, EpisodeShardWriter
    roots = [source] if (source / 'manifest.json').is_file() else sorted(source.glob('checkpoint_*'))
    roots = [root for root in roots if (root / 'manifest.json').is_file()]
    if not roots:
        raise ValueError(f'No collected datasets found in {source}')
    metadata = json.loads((roots[0] / 'metadata.json').read_text())
    keys = ('state_schema', 'action_schema', 'event_names', 'skill_policy_metadata',
            'macro_action_steps', 'high_level_dt')
    for root in roots:
        other = json.loads((root / 'metadata.json').read_text())
        for key in keys:
            if other.get(key) != metadata.get(key):
                raise ValueError(f'Incompatible {key} in {root}')
    if output.exists():
        raise ValueError(f'Prepared dataset already exists: {output}; choose a new output directory')
    metadata['purpose'] = 'offline_world_model_training'
    metadata['source_datasets'] = [str(root.resolve()) for root in roots]
    writer = EpisodeShardWriter(output, metadata)
    seen = set()
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        for entry in manifest['episodes']:
            path = root / entry['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Shard checksum mismatch: {path}')
            with np.load(path, allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in shard.files}
            digest = hashlib.sha256()
            for key in sorted(set(arrays) - {'episode_id'}):
                digest.update(key.encode())
                digest.update(arrays[key].tobytes())
            if digest.hexdigest() in seen:
                continue
            seen.add(digest.hexdigest())
            episode_id = len(writer.entries)
            arrays['episode_id'] = np.full_like(arrays['episode_id'], episode_id)
            writer.write_episode(Episode(episode_id, arrays))
    entries = list(writer.entries)
    if len(entries) < 3:
        raise ValueError('Need at least 3 distinct complete episodes for train/validation/test; collect more data')
    import random
    random.Random(seed).shuffle(entries)
    heldout = max(1, int(len(entries) * .1))
    groups = {'train': entries[2*heldout:], 'validation': entries[:heldout],
              'test': entries[heldout:2*heldout]}
    for name, selected in groups.items():
        (output / f'{name}_manifest.json').write_text(json.dumps({
            'format': 'dribblebot_world_model_v1', 'split': name, 'seed': seed,
            'episodes': selected}, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default='configs/world_model_as2.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    from quadruped.world_model.config import load_config
    from quadruped.world_model.dataset import WorldModelDataset
    from quadruped.world_model.schema import StateSchema
    from quadruped.world_model.action_adapter import JointActionAdapter
    from quadruped.world_model.ensemble import WorldModelEnsemble
    from quadruped.world_model.trainer import WorldModelTrainer, fit_normalizer, seed_everything
    config = load_config(args.config)
    config['seed'] = args.seed
    config['training'].update(device=args.device, num_workers=args.num_workers)
    for option, key in ((args.epochs, 'max_epochs'), (args.batch_size, 'batch_size')):
        if option is not None:
            if option < 1:
                parser.error(f'{key} must be positive')
            config['training'][key] = option
    seed_everything(args.seed)
    prepared = args.output / 'dataset'
    if args.resume:
        metadata = json.loads((prepared / 'metadata.json').read_text())
    else:
        if (args.output / 'best.pt').exists():
            parser.error('Output already contains a model; use --resume or a new output')
        metadata = prepare_dataset(args.dataset, prepared, args.seed)
    skills = metadata.get('skill_policy_metadata', {})
    hashes = {name: {key: artifact.get('sha256') for key, artifact in
                    record.get('artifacts', {}).items()} for name, record in skills.items()}
    if not hashes or any(not artifacts.get('body') for artifacts in hashes.values()):
        raise ValueError('Dataset lacks hashed skill provenance; recollect with the current collector')
    config['skill_fingerprint'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    config['world_model']['macro_action_steps'] = metadata['macro_action_steps']
    config['offline_dataset'] = str(prepared.resolve())
    train, validation = WorldModelDataset(prepared, 'train'), WorldModelDataset(prepared, 'validation')
    if not len(train) or not len(validation):
        raise ValueError('Training and validation splits must both be nonempty')
    schema = StateSchema.from_dict(metadata['state_schema'])
    actions = JointActionAdapter.from_dict(metadata['action_schema'])
    normalizer = fit_normalizer(train, schema, config['training'].get('normalize_reward', True))
    model_config = dict(config['model'], num_events=len(metadata['event_names']))
    model = WorldModelEnsemble(schema, actions, normalizer, **model_config)
    model.event_names = tuple(metadata['event_names'])
    if args.resume:
        import torch
        checkpoint = torch.load(args.resume, map_location='cpu')
        if checkpoint['state_schema'] != schema.to_dict() or checkpoint['action_schema'] != actions.to_dict():
            raise ValueError('Resume checkpoint does not match prepared dataset schemas')
        if checkpoint.get('training_config', {}).get('offline_dataset') != str(prepared.resolve()):
            raise ValueError('Resume must use the original output directory and prepared dataset')
        if int(checkpoint['epoch']) + 1 >= int(config['training']['max_epochs']):
            parser.error('--epochs must exceed the completed epoch count when resuming')
    trainer = WorldModelTrainer(model, train, validation, config)
    trainer.fit(args.output, resume=args.resume)
    print(f'Offline model: {args.output / "best.pt"}; held-out dataset: {prepared}')


if __name__ == '__main__':
    main()
