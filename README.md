# Visual-Attention-For-Medical-Image-Classification


### Model Architecture
The model architecture is shown as below:

<img src = https://github.com/GuoshenLi/Zoom-in-Lesion-For-Medical-Image-Classification/blob/main/model.png width = '915' height = '422'/><br/>



### The Attention Heatmap
The attention heatmap is shown as below:

<img src = https://github.com/GuoshenLi/Zoom-in-Lesion-For-Medical-Image-Classification/blob/main/heatmap.png width = '1146' height = '400'/><br/>



Explanation of the heatmap:
The atten1 is the output of the first branch of the network, which is used to zoom in the lesion of the second branch.
The input1 is the original image. The input2 in the image that zoomed in by the heatmap of attn1. We can clearly see that the lesion part is zoomed in.
The following mid1, mid2, mid3 is the output of the non-local block of the first branch. We use the att1 to zoom them in and we get mid1_zoom, mid2_zoom, mid3_zoom and concat them into the second branch respectively. 
We can get a better result in the second branch and further improve the accuracy.


