import os
import tensorflow as tf


MODEL = r"D:\MinifasnetV1\tensorflow\2.7_80x80_MiniFASNetV2_int8.tflite"


interpreter = tf.lite.Interpreter(
    model_path=MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\n==========================================")
print("INT8 MODEL INSPECTION")
print("==========================================")


# --------------------------------------------------
# Input
# --------------------------------------------------

print("\nINPUT")
print("------------------------------------------")

print("Name:")
print(input_details[0]["name"])

print("Shape:")
print(input_details[0]["shape"])

print("Dtype:")
print(input_details[0]["dtype"])

print("Quantization:")
print(input_details[0]["quantization"])

print("Quantization parameters:")
print(input_details[0]["quantization_parameters"])


# --------------------------------------------------
# Output
# --------------------------------------------------

print("\nOUTPUT")
print("------------------------------------------")

print("Name:")
print(output_details[0]["name"])

print("Shape:")
print(output_details[0]["shape"])

print("Dtype:")
print(output_details[0]["dtype"])

print("Quantization:")
print(output_details[0]["quantization"])

print("Quantization parameters:")
print(output_details[0]["quantization_parameters"])


# --------------------------------------------------
# Model size
# --------------------------------------------------

size_bytes = os.path.getsize(MODEL)

print("\nMODEL SIZE")
print("------------------------------------------")

print(f"Bytes : {size_bytes}")
print(f"KB    : {size_bytes / 1024:.2f}")
print(f"MB    : {size_bytes / (1024 * 1024):.2f}")