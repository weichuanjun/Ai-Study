import mlflow
# 各種パスを指定
TRACKING_URI = '[トラッキングサーバ（バックエンド）に指定したいパス]'
ARTIFACT_LOCATION = '[Artifactストレージに指定したいパス]'
EXPERIMENT_NAME = '[指定したいエクスペリメント名]'

mlflow.set_tracking_uri(TRACKING_URI)
# Experimentの生成
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:  # 当該Experiment存在しないとき、新たに作成
    experiment_id = mlflow.create_experiment(
                            name=EXPERIMENT_NAME,
                            artifact_location=ARTIFACT_LOCATION)
else: # 当該Experiment存在するとき、IDを取得
    experiment_id = experiment.experiment_id

