import wandb
import os

def download_weights():
    print("Downloading weights...")
    directory = ('.')
    if not os.path.exists(directory):
        os.makedirs(directory)
    run = wandb.init()
    artifact = run.use_artifact('babycar27/iglu-checkpoints/nlp-iglu-checkpoints:v0', type='pt')
    artifact_dir = artifact.download(
        root='.')
    print("Weights path - ", artifact_dir)
    
if __name__=="__main__":
    download_weights()
