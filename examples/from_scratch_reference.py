"""See every LUT update using the pure-Python reference implementation."""
from addernet import ReferenceAdderNetLayer

trace = []
layer = ReferenceAdderNetLayer(size=16, input_min=0, input_max=10, lr=1.0)
layer.train([0, 5, 10], [0, 10, 20], epochs_raw=2, epochs_expanded=3,
            trace=lambda event: trace.append(event))

print("first five updates:")
for event in trace[:5]:
    print(event)
print("prediction for 7:", layer.predict(7))
print("table:", layer.offset_table)
