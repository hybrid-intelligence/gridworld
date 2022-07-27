import wandb

run = wandb.init(name = 'babycar27', project = 'iglu-checkpoints', job_type='train')
artifact = wandb.Artifact('nlp-iglu-checkpoints', type='pt')
artifact.add_file('t5-autoregressive-history-3-best.pt')

run.log_artifact(artifact)

# artifact = run.use_artifact('iglu-checkpoints:v0')
#
# # Download the artifact's contents
# artifact_dir = artifact.download(root='train_dir/0012/force_envs_single_thread=False;num_envs_per_worker=1;num_workers=10/'+
#                    'TreeChopBaseline-iglu/checkpoint_p0/')
# print(artifact_dir)
