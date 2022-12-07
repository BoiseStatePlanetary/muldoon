# Muldoon

## Members:
 - Jim Crowell
 - Samantha Farmer
 - Lukas Karlsson
 - Senami Hodonu

## Abstract:
To support the capability of quantifying Martian dust devils atmospheric influence on Mars's climate, we plan to implement and improve the open source Muldoon python package. This package already has a model in place and our goal is to improve the efficiency of this existing algorithm and create more algorithms for the remaining measurements to inform us of the impact dust devils have on Mars. The models we will be implementing are the wind time series, dust signal time series. Our main goals for this project will be to add PDS4 compatibility (format used by NASA to store and distribute planetary data), improve the packages documentation, and implement wind speed time series model to the Muldoon Python package. This work could lead to Muldoon being a helpful tool for other scientists and engineers who have to analyze rover data and the effects of dust devils on Mars’s climate.

## Project Description:
The goal of project was adapting and upgrading a previously constructed Python package: Muldoon, which is designed to analyze Martian meteorological data. Planned upgrades included:  
- Reconfiguring python package to read PDS4 data  
- Implementing a dust devil wind timeseries model  
- Improving the package documentation  

A user of this Python package would most likely be well versed in this kind of research, so that would have to be kept in mind while viewing examples of the following material; it is understood that as esoteric as meteorology already is, even before you factor in the literally-alien environment that is the target, it can be a little overwhelming to the layperson.

## Screen Shots:
![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/code_ex.png)  
- The simplest drivers of the program, which read data either from a remote repository or local file, process it, find appropriate labels and headers in the data for plotting, and create a plot, which is customizable depending on optional arguments supplied  

![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/output.png)  
- A plot showing oscillations in horizontal wind speed  

![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/output1.png)  
- Isolated pressure changes over time. Note the linear regression in the center, which is a giveaway for dust devil activity  

![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/output3.png)  
- Isolated temperature oscilations over time. It is worth noting that these are graphs a researcher would make use of to analyze individual aspects of the dust devil, once you've already found one  

![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/code_ex2.png)  
- An excerpt of the wind timeseries code, which specifically finds the metrics behind an individual dust devil's profile  

![img](https://github.com/cs481-ekh/f22-dust-devils/blob/main/docs/output4.png)  
- The plot created by the code above, which shows a graph representation of a dust devil passing over/through the Rover
