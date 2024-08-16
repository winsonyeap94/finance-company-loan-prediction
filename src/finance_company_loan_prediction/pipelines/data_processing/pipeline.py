from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_dataset,
                inputs={"data_df": "train_dataset"},
                outputs=["preprocessed_train_dataset", "knn_imputer"],
                name="preprocess_train_dataset"
            ),
            node(
                func=preprocess_dataset,
                inputs={"data_df": "test_dataset", "reference_df": "preprocessed_train_dataset", "imputer": "knn_imputer"},
                outputs=["preprocessed_test_dataset", "_"],
                name="preprocess_test_dataset"
            )
        ]
    )