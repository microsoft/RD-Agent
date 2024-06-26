from pydantic_settings import BaseSettings


class ModelPropSetting(BaseSettings):
    """"""

    scen: str  # a.b.c:XXXClass
    # TODO: inital keywards should be included in the settings
    ...


MODEL_PROP_SETTING = ModelPropSetting()
