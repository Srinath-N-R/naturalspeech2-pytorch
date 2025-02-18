import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from naturalspeech2_pytorch.naturalspeech2_pytorch import (
    NaturalSpeech2,
    Wavenet,
    Model,
    CustomDataset,
    PhonemeEncoder,
    DurationPitchPred,
    SpeechPromptEncoder,
    Tokenizer,
    ESpeak,
    custom_collate_fn
)

from audiolm_pytorch import (
    SoundStream,
    EncodecWrapper
)
