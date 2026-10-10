"""7장 실제 KFP v2 파이프라인(kfp 2.17.0 SDK 문법). 이 실습 저장소의 테스트는 이 파일을 import하지 않는다.

이 저장소의 의존성(numpy, torch, pytest)에는 kfp가 없다. 이 파일은 kfp 2.17.0을 따로 설치한 환경에서
kfp_local_check.py가 컴파일(IR YAML)하고 kfp.local로 실제 실행해 본 것이다. 컴파일 결과는
results/kfp_ir_2.17.0.yaml(원본)과 results/kfp_ir_2.17.0.json(같은 내용을 JSON으로 바꾼 것)에 있고,
test_pipelines_ch07_kfp.py는 이 JSON과 우리 최소 구현(pipelines_ch07_dag.py)의 컴파일 결과가 같은 그래프인지 비교한다.

컴포넌트 몸통은 hermetic(함수 밖의 이름을 쓰지 않음) 규칙을 따른다. 몸통 안에서 pipelines_ch07_steps를
import하므로, 클러스터에서는 그 모듈이 든 이미지(4·5장에서 만든 이미지, 커밋 SHA 태그)를 base_image로 줘야 한다.
"""

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

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


if __name__ == "__main__":
    compiler.Compiler().compile(churn_pipeline, package_path="churn_ch07.yaml")
