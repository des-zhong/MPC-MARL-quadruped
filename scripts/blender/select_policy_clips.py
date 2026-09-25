"""Select independent, reset-free real match excerpts and resample to 24 FPS."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=36);p.add_argument('--seconds',type=float,default=10);p.add_argument('--recording',type=Path);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);recording=a.recording or out
z=np.load(recording/'rollout.npz');meta=json.loads((recording/'rollout.json').read_text())
links=z['links'];balls=z['ball'];ep=z['episode_ids'];dt=meta['dt'];fps=24;seconds=a.seconds;count=round(fps*seconds)
assert count>1 and abs(count/fps-seconds)<1e-6
span=int(np.ceil((count-1)/fps/dt))+1
candidates=[]
for env in range(ep.shape[1]):
    cuts=np.r_[0,np.flatnonzero(np.diff(ep[:,env]))+1,len(ep)]
    best=None
    for start,end in zip(cuts[:-1],cuts[1:]):
        # Skip settling after reset; don't include reset or terminal frames.
        for s in range(start+20,end-span-2,10):
            b=balls[s:s+span,env,:3];r=links[s:s+span,env,:,0,:3]
            if np.max(np.abs(b[:,:2]),axis=0)[0]>4.05 or np.max(np.abs(b[:,1]))>2.5:continue
            travel=float(np.linalg.norm(np.diff(b[:,:2],axis=0),axis=1).sum())
            activity=float(np.linalg.norm(np.diff(r[:,:,:2],axis=0),axis=2).sum())
            score=travel+.10*activity
            entry={'source_env':env,'source_episode':int(ep[s,env]),'start_sample':s,'source_start_s':s*dt,'ball_travel_m':travel,'robot_travel_m':activity,'score':score}
            if best is None or score>best['score']:best=entry
    if best:candidates.append(best)
candidates.sort(key=lambda c:c['score'],reverse=True)
assert len(candidates)>=a.num_envs,f'Only {len(candidates)} independent {seconds:g}-second excerpts available'
selected=candidates[:a.num_envs]
# Position linear interpolation; rotations normalized shortest-arc SLERP.
def interp(data,idx,w):
    x=data[idx];y=data[idx+1];q0=x[...,3:7];q1=y[...,3:7]
    dot=np.sum(q0*q1,axis=-1,keepdims=True);q1=np.where(dot<0,-q1,q1);dot=np.clip(np.abs(dot),0,1)
    theta=np.arccos(dot);den=np.maximum(np.sin(theta),1e-8)
    wb=w.reshape((len(w),)+(1,)*(data.ndim-1))
    q=np.sin((1-wb)*theta)/den*q0+np.sin(wb*theta)/den*q1
    q=np.where(dot>.9995,(1-wb)*q0+wb*q1,q);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    return np.concatenate(((1-wb)*x[...,:3]+wb*y[...,:3],q),axis=-1)
track=[];balltrack=[]
for c in selected:
    env=c['source_env'];samples=c['start_sample']+np.arange(count)/fps/dt;idx=np.floor(samples).astype(int);w=samples-idx
    assert np.all(ep[idx,env]==c['source_episode']) and np.all(ep[idx+1,env]==c['source_episode'])
    track.append(interp(links[:,env],idx,w));balltrack.append(interp(balls[:,env],idx,w))
track=np.stack(track,axis=1);balltrack=np.stack(balltrack,axis=1)
np.savez_compressed(out/'clips_24fps.npz',links=track,ball=balltrack,body_names=z['body_names'])
(out/'clip_selection.json').write_text(json.dumps({'fps':fps,'frames':count,'duration_s':seconds,'candidates':len(candidates),'recording':str(recording.resolve()),'description':f'{a.num_envs} distinct simulated environments, each independently trimmed to a continuous {seconds:g}-second excerpt; no time scaling or repeated robot animations','clips':selected},indent=2))
print('SELECTED',len(selected),'from',len(candidates),'best',selected[0],'shapes',track.shape,balltrack.shape)

# Verify the baked tracks, including fixed calf-to-foot attachments.
names=z['body_names'].tolist();errors={}
for leg in ('FL','FR','RL','RR'):
    c=track[:,:,:,names.index(leg+'_calf')];f=track[:,:,:,names.index(leg+'_foot')]
    q=c[...,3:7];v=np.zeros_like(c[...,:3]);v[...,2]=-.21344
    t=2*np.cross(q[...,:3],v);rot=v+q[...,3:4]*t+np.cross(q[...,:3],t)
    errors[leg]=float(np.max(np.linalg.norm(c[...,:3]+rot-f[...,:3],axis=-1)))
qerror=float(np.max(np.abs(np.linalg.norm(track[...,3:7],axis=-1)-1)))
travel=np.linalg.norm(np.diff(track[:,:,:,0,:2],axis=0),axis=-1).sum(axis=0)
assert max(errors.values())<.005 and qerror<1e-5 and np.isfinite(track).all()
(out/'motion_validation.json').write_text(json.dumps({
    'max_fixed_foot_attachment_errors_m':errors,'attachment_check_tolerance_m':.005,
    'attachment_note':'Independent world-link interpolation can introduce millimetre-scale fixed-joint deviations.',
    'finite_poses':bool(np.isfinite(track).all()),'max_quaternion_norm_error':qerror,
    'frames':len(track),'robot_link_tracks':int(np.prod(track.shape[1:4])),
    'min_robot_base_travel_m':float(travel.min()),'max_robot_base_travel_m':float(travel.max())},indent=2))
