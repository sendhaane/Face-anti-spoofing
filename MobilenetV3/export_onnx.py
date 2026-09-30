import torch

from src.mobilenetv3 import mobilenetv3_large


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/MN3_antispoof.pth"
ONNX_PATH = "models/MN3_antispoof.onnx"

DEVICE = "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading MobileNetV3...")

device = torch.device(DEVICE)

model = mobilenetv3_large(
    width_mult=1.0,
    prob_dropout=0.1,
    type_dropout="bernoulli",
    prob_dropout_linear=0.35,
    embeding_dim=1280,
    mu=0.5,
    sigma=0.3,
    theta=0,
    multi_heads=True
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

state_dict = checkpoint["state_dict"]

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# WRAPPER
# ============================================================

class AntiSpoofONNX(torch.nn.Module):

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):

        # Extract MobileNetV3 features
        features = self.model(x)

        # Main anti-spoofing head
        logits = self.model.make_logits(
            features,
            all=False
        )

        # In case make_logits returns a tuple
        if isinstance(logits, tuple):
            logits = logits[0]

        return logits


onnx_model = AntiSpoofONNX(model)
onnx_model.eval()


# ============================================================
# DUMMY INPUT
# ============================================================

dummy_input = torch.randn(
    1,
    3,
    128,
    128,
    dtype=torch.float32
).to(device)


# ============================================================
# TEST PYTORCH OUTPUT
# ============================================================

with torch.no_grad():

    pytorch_output = onnx_model(
        dummy_input
    )

print()
print("PyTorch output shape:")
print(pytorch_output.shape)

print()
print("PyTorch output:")
print(pytorch_output)


# ============================================================
# EXPORT ONNX
# ============================================================

print()
print("Exporting ONNX model...")

torch.onnx.export(
    onnx_model,
    dummy_input,
    ONNX_PATH,

    export_params=True,

    opset_version=17,

    do_constant_folding=True,

    input_names=[
        "input"
    ],

    output_names=[
        "logits"
    ],

    dynamic_axes={
        "input": {
            0: "batch"
        },
        "logits": {
            0: "batch"
        }
    }
)


print()
print("=" * 60)
print("ONNX EXPORT COMPLETE")
print("=" * 60)

print()
print(f"ONNX model: {ONNX_PATH}")
print(f"Input shape:  [batch, 3, 128, 128]")
print(f"Output shape: [batch, 2]")