import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("../training_rcnn_images/training_predictions.csv")
# df = pd.read_csv("../crowd_predictions/predictions.csv")
# print(df)

# fig = px.scatter(df, x="Mask", y="YOLO_Person")
# fig = px.scatter(df, x="Mask", y="RCNN_Person")
# fig.update_layout(title="Predicted Mask vs. Without Mask Count Bargraph (800 images)")
# fig.show()
# df['average_compliance'] = (df['Minimum_Compliance'] + df['Maximum_Compliance']) / 2
# df['acceptable_compliance'] = np.where(df['average_compliance'] <= 0.5, "Less Than or Meets Standard (50%)", "Better than standard (50%)")
# fig = px.bar(df, x="average_compliance", y=["Sample" for i in df['average_compliance']],color='acceptable_compliance', orientation='h',
#              height=400,
#              title='Compliance vs. Number of People (800 People)')
# fig.show()

{'without_mask': 89, 'with_mask': 23}​
{'with_mask': 97, 'without_mask': 1}​


# fig = px.scatter(df, x="Mask", y="Without_Mask")
# fig.update_layout(title="Predicted Mask vs. Without Mask Count Scatterplot (800 images)")
# fig.show()

# import plotly.graph_objects as go


# fig = go.Figure()
# fig.add_trace(go.Box(y=df['Minimum_Compliance'],name="Minimum Predicted Compliance"))
# fig.add_trace(go.Box(y=df['Maximum_Compliance'],name="Maximum Predicted Compliance"))
# fig.update_traces(boxpoints='all', jitter=0)
# fig.update_layout(title="Predicted Mask Compliance from Sample of 800 images (Training Set)")
# fig.show()

# Mask  Incorrect_Mask  Without_Mask
# sum_masks = sum(df["Mask"])
# sum_incorrect_masks = sum(df["Incorrect_Mask"])
# sum_no_masks = sum(df["Without_Mask"])
# plt.bar(["Without_Mask","Incorrect_Mask","Mask"],[sum_no_masks,sum_incorrect_masks,sum_masks])
# plt.show()





