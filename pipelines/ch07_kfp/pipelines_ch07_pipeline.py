"""7장 파이프라인 정의(최소 구현판). kfp_pipeline.py와 같은 파이프라인을 pipelines_ch07_dag로 적었다.

맨 위 import 줄만 다르다. pipelines_ch07_dag를 dsl이라는 이름으로 불러오므로, 아래 데코레이터·시그니처·몸통은
kfp_pipeline.py와 글자까지 같다(test_kfp_file_and_mini_file_define_the_same_functions가 함수 전체를 비교한다).
"""

import pipelines_ch07_dag as dsl
from pipelines_ch07_dag import Dataset, Input, Metrics, Model, Output

IMAGE = "registry.example.com/churn:GIT_SHA"  # Jenkins가 커밋 SHA로 바꿔 넣는다(Jenkinsfile 참고)


@dsl.component(base_image=IMAGE)
def prepare(seed: int, n_rows: int, shift: float, train_data: Output[Dataset], valid_data: Output[Dataset]):
    import pipelines_ch07_steps as S
    S.prepare(seed, n_rows, shift, train_data.path, valid_data.path)


@dsl.component(base_image=IMAGE)
def train(train_data: Input[Dataset], model: Output[Model], impl: str = "numpy"):
    import pipelines_ch07_steps as S
    info = S.train(train_data.path, model.path, impl)
    model.metadata.update(info)


@dsl.component(base_image=IMAGE)
def evaluate(model: Input[Model], valid_data: Input[Dataset], metrics: Output[Metrics], threshold: float = 0.5):
    import pipelines_ch07_steps as S
    m = S.evaluate(model.path, valid_data.path, metrics.path + ".json", threshold)
    metrics.log_metric("accuracy", m["accuracy"])


@dsl.pipeline(name="churn-ch07")
def churn_pipeline(seed: int = 1, n_rows: int = 4000, shift: float = 1.0, impl: str = "numpy", threshold: float = 0.5):
    p = prepare(seed=seed, n_rows=n_rows, shift=shift)
    t = train(train_data=p.outputs["train_data"], impl=impl)
    evaluate(model=t.outputs["model"], valid_data=p.outputs["valid_data"], threshold=threshold)
