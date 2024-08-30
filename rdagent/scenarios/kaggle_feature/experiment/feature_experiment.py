import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.coder.feature_coder.config import FEATURE_IMPLEMENT_SETTINGS
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle_feature.experiment.workspace import KGFFBWorkspace
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions
from rdagent.utils.env import KGDockerConf


prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGFeatureExperiment(FactorExperiment[FactorTask, KGFFBWorkspace, KGFFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFFBWorkspace(template_folder_path=Path(__file__).parent.parent.parent/"kaggle/experiment/meta_tpl")


class KGFeatureScenario(Scenario):
    def __init__(self, competition: str) -> None:
        super().__init__()
        self.competition = competition
        self.competition_descriptions = crawl_descriptions(competition)
        self.competition_type = None
        self.competition_description = None
        self.target_description = None
        self.competition_features = None
        self._analysis_competition_description()

    def _analysis_competition_description(self):

        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_description_template"]["system"])
            .render()
        )

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_description_template"]["user"])
            .render(
                competition_descriptions=self.competition_descriptions,
            )
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json_analysis = json.loads(response_analysis)
        self.competition_type = response_json_analysis.get("Competition Type", "No type provided")
        self.competition_description = response_json_analysis.get("Competition Description", "No description provided")
        self.target_description = response_json_analysis.get("Target Description", "No target provided")
        self.competition_features = response_json_analysis.get("Competition Features", "No features provided")

        X_train, X_valid, y_train, y_valid, X_test, passenger_ids = self.preprocess_script()
        self.competition_features = X_train.columns.tolist()
        self.competition_features = X_train.head()

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_feature_background"]

        background_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(background_template)
            .render(
                competition_type=self.competition_type,
                competition_description=self.competition_description,
                target_description=self.target_description,
                competition_features=self.competition_features,
            )
        )

        return background_prompt

    @property
    def source_data(self) -> pd.DataFrame:
        # #TODO: Implement source_data
        # kaggle_conf = KGDockerConf()
        # data_path = Path(f"{kaggle_conf.share_data_path}/{self.competition}")
        # file_path = data_path / "train.csv"
        # data = pd.read_csv(file_path)
        #TODO later we should improve this part
        data_folder = Path(FEATURE_IMPLEMENT_SETTINGS.data_folder)

        if (data_folder / "train.csv").exists():
            X_train = pd.read_csv(data_folder / "train.csv")
            # X_valid = pd.read_csv(data_folder / "valid.csv")
            # X_test = pd.read_csv(data_folder / "test.csv")
            return X_train.head()
        
        X_train, X_valid, y_train, y_valid, X_test, passenger_ids = self.preprocess_script()

        data_folder.mkdir(exist_ok=True, parents=True)
        X_train.to_csv(data_folder / "train.csv", index=False)
        X_valid.to_csv(data_folder / "valid.csv", index=False)
        X_test.to_csv(data_folder / "test.csv", index=False)
        return X_train.head()
        raise NotImplementedError("source_data is not implemented")

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_feature_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["kg_feature_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["kg_feature_simulator"]

    @property
    def rich_style_description(self) -> str:
        return """
kaggle scen """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""
    def prepreprocess(self):
        """
        This method loads the data, drops the unnecessary columns, and splits it into train and validation sets.
        """
        # Load and preprocess the data
        data_df = pd.read_csv("/home/v-xisenwang/git_ignore_folder/data/playground-series-s4e8/train.csv")
        data_df = data_df.drop(["id"], axis=1)

        X = data_df.drop(["class"], axis=1)
        y = data_df[["class"]]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)  # Convert class labels to numeric

        # Split the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)

        return X_train, X_valid, y_train, y_valid

    def preprocess(self, X: pd.DataFrame):
        """
        Preprocesses the given DataFrame by transforming categorical and numerical features.
        Ensures the processed data has consistent features across train, validation, and test sets.
        """

        # Identify numerical and categorical features
        numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]
        categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

        # Define preprocessors for numerical and categorical features
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_cols),
                ("num", numerical_transformer, numerical_cols),
            ]
        )

        # Fit the preprocessor on the data and transform it
        preprocessor.fit(X)
        X_array = preprocessor.transform(X).toarray()

        # Get feature names for the columns in the transformed data
        feature_names = (
            preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols).tolist()
            + numerical_cols
        )

        # Convert arrays back to DataFrames
        X_transformed = pd.DataFrame(X_array, columns=feature_names, index=X.index)

        return X_transformed

    def preprocess_script(self):
        """
        This method applies the preprocessing steps to the training, validation, and test datasets.
        """
        X_train, X_valid, y_train, y_valid = self.prepreprocess()

        # Preprocess the train and validation data
        X_train = self.preprocess(X_train)
        X_valid = self.preprocess(X_valid)

        # Load and preprocess the test data
        submission_df = pd.read_csv("/home/v-xisenwang/git_ignore_folder/data/playground-series-s4e8/test.csv")
        passenger_ids = submission_df["id"]
        submission_df = submission_df.drop(["id"], axis=1)
        X_test = self.preprocess(submission_df)

        return X_train, X_valid, y_train, y_valid, X_test, passenger_ids