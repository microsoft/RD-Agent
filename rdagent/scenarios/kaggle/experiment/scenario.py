import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions
from rdagent.utils.env import KGDockerConf

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGModelScenario(Scenario):
    def __init__(self, competition: str) -> None:
        super().__init__()
        self.competition = competition
        self.competition_descriptions = crawl_descriptions(competition)
        self.competition_type = None
        self.competition_description = None
        self.target_description = None
        self.competition_features = None
        self._analysis_competition_description()
        self._background = self.background
        self._source_data = self.source_data
        self._output_format = self.output_format
        self._interface = self.interface
        self._simulator = self.simulator

    def _analysis_competition_description(self):
        # TODO: use GPT to analyze the competition description

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

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_background"]

        train_script = (Path(__file__).parent / "meta_tpl" / "train.py").read_text()

        background_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(background_template)
            .render(
                train_script=train_script,
                competition_type=self.competition_type,
                competition_description=self.competition_description,
                target_description=self.target_description,
                competition_features=self.competition_features,
            )
        )
        return background_prompt

    @property
    def source_data(self) -> str:
        kaggle_conf = KGDockerConf()
        data_path = Path(f"{kaggle_conf.share_data_path}/{self.competition}")

        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            return "No CSV files found in the specified path."

        dataset = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

        simple_eda = dataset.info(buf=None)  # Capture the info output
        data_shape = dataset.shape
        data_head = dataset.head()

        eda = (
            f"Basic Info about the data:\n{simple_eda}\n"
            f"Shape of the dataset: {data_shape}\n"
            f"Sample Data:\n{data_head}\n"
        )

        data_description = self.competition_descriptions.get("Data Description", "No description provided")
        eda += f"\nData Description:\n{data_description}"

        return eda

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_model_output_format"]

    @property
    def interface(self) -> str:
        return f"""The feature code should follow the interface:
{prompt_dict['kg_feature_interface']}
The model code should follow the interface:
{prompt_dict['kg_model_interface']}
"""

    @property
    def simulator(self) -> str:
        return prompt_dict["kg_model_simulator"]

    @property
    def rich_style_description(self) -> str:
        return """
kaggle scen """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self._background}
The interface you should follow to write the runnable code:
{self._interface}
The output of your code should be in the format:
{self._output_format}
The simulator user can use to test your model:
{self._simulator}
"""

    def prepreprocess(self):
        """
        This method loads the data, drops the unnecessary columns, and splits it into train and validation sets.
        """
        # Load and preprocess the dataoi/i8p89oiu7g;o87;87;c8
        data_df = pd.read_csv("git_ignore_folder/data/playground-series-s4e8/train.csv")
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
        submission_df = pd.read_csv("git_ignore_folder/data/playground-series-s4e8/test.csv")
        passenger_ids = submission_df["id"]
        submission_df = submission_df.drop(["id"], axis=1)
        X_test = self.preprocess(submission_df)

        return X_train, X_valid, y_train, y_valid, X_test, passenger_ids
