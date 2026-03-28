from inference import AIImageDetectorNN

# Test spatial only
spatial_detector = AIImageDetectorNN(
    spatial_weights='checkpoints/spatial_model_best.pth'
)

# Test frequency only
frequency_detector = AIImageDetectorNN(
    frequency_weights='checkpoints/frequency_model_best.pth'
)

image = r"C:\Users\User\Downloads\download (6).jpg"

s_result = spatial_detector.detect(image)
f_result = frequency_detector.detect(image)

print("=== SPATIAL MODEL ===")
print(f"Verdict : {s_result['spatial']['verdict']}")
print(f"Real    : {s_result['spatial']['real_probability']}%")
print(f"AI      : {s_result['spatial']['ai_probability']}%")

print("\n=== FREQUENCY MODEL ===")
print(f"Verdict : {f_result['frequency']['verdict']}")
print(f"Real    : {f_result['frequency']['real_probability']}%")
print(f"AI      : {f_result['frequency']['ai_probability']}%")