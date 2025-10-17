import fire
import torch
import torch.nn as nn
from ..utils.gpu_utils import setup_gpu
from rdagent.components.coder.model_coder.task_loader import (
    ModelExperimentLoaderFromPDFfiles,
)
from rdagent.components.document_reader.document_reader import (
    extract_first_page_screenshot_from_pdf,
)
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.qlib.developer.model_coder import QlibModelCoSTEER

class GPUEnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GPUEnhancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = setup_gpu()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden states on correct device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesModelFactory:
    def create_model(self, model_type, **kwargs):
        model = None
        if model_type == "lstm":
            model = GPUEnhancedLSTM(
                input_size=kwargs.get('input_size', 10),
                hidden_size=kwargs.get('hidden_size', 50),
                num_layers=kwargs.get('num_layers', 2),
                output_size=kwargs.get('output_size', 1)
            )
        if model:
            model = model.to(setup_gpu())
        return model
    
def extract_models_and_implement(report_file_path: str) -> None:
    """
    This is a research copilot to automatically implement models from a report file or paper.

    It extracts models from a given PDF report file and implements the necessary operations.

    Parameters:
    report_file_path (str): The path to the report file. The file must be a PDF file.

    Example URLs of PDF reports:
    - https://arxiv.org/pdf/2210.09789
    - https://arxiv.org/pdf/2305.10498
    - https://arxiv.org/pdf/2110.14446
    - https://arxiv.org/pdf/2205.12454
    - https://arxiv.org/pdf/2210.16518

    Returns:
    None
    """
    scenario = GeneralModelScenario()
    logger.log_object(scenario, tag="scenario")
    # Save Relevant Images
    img = extract_first_page_screenshot_from_pdf(report_file_path)
    logger.log_object(img, tag="pdf_image")
    exp = ModelExperimentLoaderFromPDFfiles().load(report_file_path)
    logger.log_object(exp, tag="load_experiment")
    exp = QlibModelCoSTEER(scenario).develop(exp)
    logger.log_object(exp, tag="developed_experiment")


if __name__ == "__main__":
    fire.Fire(extract_models_and_implement)
