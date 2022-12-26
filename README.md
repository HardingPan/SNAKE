####################  
51单片机课设上位机部分  
作者：Harding  
日期：2022.12  
####################  
  
  
【文件说明】  
imageProcessing.py主要负责图像的预处理  
fourierDescriptor.py主要负责特征的提取  
  
【Python库说明】  
opencv-python，numpy，matplotlib，  
（以上库的版本没有具体要求）  
  
  
【图像处理具体信息】  
主要步骤为：  
1，滤波去噪（感觉可有可无）  
2，皮肤检测（实际使用中发现skin_ellipse函数的效果更好）   
4，形态学处理（简单的开运算，但是没有达到较好的效果，还需要后续再处理）   
   
【特征提取具体信息】   
参考连接：https://github.com/timfeirg/Fourier-Descriptors   

  