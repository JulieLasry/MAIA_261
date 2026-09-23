### MAIA 261 - 2D Chest X-Ray classification
## Notebooks README

# Scope
The notebooks folder only contains the data_exploration 
juypiter notebook. Indeed, this data exploration part was
done in a notebook to easily and quickly understand the 
files, their content, how to preprocess them and the main
difficulties. 

# Commands
To run the notebook cells, we wan either click on the Run All
button, on top of the notebook file, or on each the individual
cells Run button.   

# Steps
- Step 1 : Data loading and checking 
- Step 2 : NA values, labels and duplication checking
- Step 3 : CSV filenames and images matching with metadata checking
- Step 4 : Class imbalance 
- Step 5 : Random samples per category visualization
- Step 6 : Image format, dtype and photometry
- Stpe 7 : Pixel spacing checking
- Step 8 : Pixel intensities inspection

# Steps details
- Stpe 1 : 
Loading the data, checking the number of samples with the number 
of rows and columns, and corrupted files inspection.

- Step 2 : 
Analyzing the NA values in our CSV, the labels different values and 
if any filename is duplicated.

- Step 3 : 
Inspecting if CSV filenames are the same as the 2D X-Ray samples and
examining the available metadata, to point out patient relative 
information (for data leakage). 

- Step 4 : 
Creating label counts plots to visualize class imbalance and printing
the exact number of samples per category.

- Step 5 : 
Visualizing 6 random images per category, to inspect le laterality, 
the image sizes and if they are cenetered.

- Step 6 : 
Inspecting the image format, photometry and dtype, while plotting the
image sizes (width and height) histograms. Information about the min,
max and mode of the images' dimensions is available. 

- Step 7 : 
Looking for the resolution (and thus pixel spacing) value in the files'
metadata. 

- Step 8 : 
Computing the pixel intensities histograms and the images mean and std
for each category, to analyze the contrast and brightness of the files. 

# Important notes 
No preprocessing methods are applied with this file. 
