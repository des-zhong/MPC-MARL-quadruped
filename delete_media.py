import wandb
api = wandb.Api()

run = api.run("des_zhong/walking/4ig572qs")

extension = ".gif"
files = run.files()
for file in files:
    print(file.name)
    if file.name.endswith(extension):
        print(file.name)
        file.delete()