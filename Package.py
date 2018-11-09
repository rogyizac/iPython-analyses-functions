
# coding: utf-8

# <h1 style="font-size:42px; text-align:center; margin-bottom:30px;">PACKAGE CODING FOR DATA ANALYSIS</h1>
# 
# <h2 style="font-size:21px; text-align:center; margin-bottom:30px;">Submitted by Shubham bansla & Rohan Isaac</h2>
# 

# <h1 style="font-size:px; text-align:LEFT; margin-bottom:px;">Importing Libraries</h1>

# In[7]:


#Operating system for changing and creating the directory
import os

# Pandas for DataFrames
import pandas as pd

# NumPy for numerical computing
import numpy as np

#ipywidgets for creatig buttons and text spaces
import ipywidgets as widgets

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Seaborn for easier visualization
import seaborn as sns


# <h1 style="font-size:px; text-align:LEFT; margin-bottom:px;">FORM 1: FUNCTION  </h1>

# In[8]:


def form1():
    print("===============Exploratory Data Analysis ==================")
    print()
    print()
    print("===============Reading The Data Set =======================")
    #widgets for directory and file for reading the data...
    style = {'description_width': '100px'}
    text_fd = widgets.Text(placeholder=r"C:\Users\shubh\Desktop\PYTHON END TERM ASSIGNMENT-2",
                           description = "Directory Name",
                          style =style,disabled=False)
    text_fn = widgets.Text(placeholder="Eg. \\cars.csv",
                           description ="File Name",
                          style =style, disabled =False)
    display(text_fd)
    display(text_fn)

    def readfile(sender):
        global load_file
        load_file = pd.read_csv( text_fd.value + '\\' + text_fn.value )
        print("Your File is loaded")

    #=== Button 1 - Reading the data set =================
    button_read = widgets.Button(description="Read Data")
    button_read.on_click(readfile)          

    #=== Button 2- Displaying some rows of the data set ===
    button_display = widgets.Button(description="Display Data")

    def displayfile(reader):
        display(load_file.head())

    button_display.on_click(displayfile) 

    #=== Button3- Displaying summaryof the data set =====
    button_summary = widgets.Button(description="Summary")

    def summary(file):
        display(load_file.describe())

    button_summary.on_click(summary)   
    display(widgets.HBox((button_read, button_display,button_summary))) # Displaying all buttons in a single row


    print("===============Suggestion No. 1=============================")

    # Step 1: - Converting all data into the numerical data
    button_conv = widgets.Button(description="Convert")

    def convert(sender):
        df1 = load_file[load_file.describe().columns]    
    print("Press convert to convert your data into numerical data.")
    button_conv.on_click(convert)
    print("")
    display(button_conv)

    # Step :2 - Taking the columns number from the user 
    print("Enter the columns you want to analyse.")
    style = {'description_width': '200px'}
    text_p1 = widgets.Text(placeholder="eg: 1,2,3",
                           description = "Enter the columns name here",
                           style = style,disabled=False)
    display(text_p1)
    
    #Create graphs for required columns and save in current working directory
    def suggestion1_func(sender):
            # In this step we are taking only numerical data
            df1 = load_file[load_file.describe().columns] 
            choice = text_p1.value.split(',')
            for i,k in enumerate(choice):
                choice[i] = int(k)
            columns=[]
            for name in choice:
                columns.append(df1.columns[name])
            #Finding categorical and numerical variables and appending to a list
            cat = []
            for t in columns:
                if((df1[t].dtype == np.float64) or (df1[t].dtype == np.int64)):
                    if df1[t].nunique() <= 15:
                        cat.append('cat')
                    else:
                        cat.append('num')
                else:
                    cat.append('cat')
            #Create histograms and barcharts
            for h, k in enumerate(columns):
                if cat[h] == 'num':
                    df1.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()
                elif df1[k].nunique() <= 15:
                    df1[k].value_counts().plot(kind = 'barh')
                    plt.xlabel('Count')
                    plt.ylabel(k)
                    plt.title('Barchart of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()


    # Step :3 - Showing graphs for only that coloumn
    button_sug = widgets.Button(description="Graphs")

    button_sug.on_click(suggestion1_func)
    display(button_sug)

    print("===============Suggestion No. 2=============================")

    print("This suggestion takes only the numerical variable when the user don't enter any coloumn.")

    # Step 1:- Taking the default input from the user
    style = {'description_width': '100px'}
    text_default = widgets.Text(placeholder="eg: all",
                           description = "Enter 'ALL' : ",
                           style = style,disabled=False)
     #### Define functions here suggestion 2 step 1 and step 2
    display(text_default)

    # Step 3:- Showing graphs for only the numerical variable as default
    print("Press the graph button for output.")
    button_default = widgets.Button(description="Graphs")
    
    #Create graphs for all variables and save them in files in current working directory
    def suggestion2_func(sender):
            df2 = load_file
            choice = text_default.value.split()
            columns=[]
            #user inputs all in text box
            if choice[0] == 'all':
                for t in df2.columns:
                    if((df2[t].dtype == np.float64) or (df2[t].dtype == np.int64)):
                        if (df2[t].nunique() >= 15) and (choice[0] == 'all'):
                            columns.append(t)
            #Find categorical and numerical variables and append to list..
            cat = []
            for t in columns:
                if((df2[t].dtype == np.float64) or (df2[t].dtype == np.int64)):
                    if (df2[t].nunique() <= 15) and (choice[0] != 'all'):
                        cat.append('cat')
                    else:
                        cat.append('num')
                elif choice[0] != 'all':
                    cat.append('cat')
            #Create graphs
            for h, k in enumerate(columns):
                print('sug2')
                if cat[h] == 'num':
                    df2.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()
                elif df[k].nunique() <= 15:
                    df2[k].value_counts().plot(kind = 'barh')
                    plt.xlabel('Count')
                    plt.ylabel(k)
                    plt.title('Barchart of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()

    button_default.on_click(suggestion2_func)
    display(button_default)

    print("===============Suggestion No. 3=============================")

    print("This suggestion takes the whole data and create graphs for every variable depending upon their types.")
    print()
    print("For numerical variable print : Histogram and Boxplot")
    print("For categorical variable print : Bar Chart")

    #Create selected graphs (histograms, bargraphs, etc as selected by user) for selected columns in current working directory
    def suggestion3_func(value):
        #passing a value to select type of graph..
        df3 = load_file
        columns = df3.columns
        cat = []
        for t in columns:
            if((df3[t].dtype == np.float64) or (df3[t].dtype == np.int64)):
                if df3[t].nunique() <= 15:
                    cat.append('cat')
                else:
                    cat.append('num')
            else:
                cat.append('cat')
        if value == 'hist':
            for h, k in enumerate(columns):
                if cat[h] == 'num':
                    df3.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()
        elif value == 'box_plot':
            for h, k in enumerate(columns):
                if cat[h] == 'num':
                    df3.boxplot(column=k)
                    plt.title('Boxplot of ' + k)
                    plt.savefig(k + '.png')
                    plt.show()          
        elif value == 'bar':
            for h, k in enumerate(columns):
                if cat[h] == 'cat':
                    if df3[k].nunique() <= 15:
                        df3[k].value_counts().plot(kind = 'barh')
                        plt.xlabel('Count')
                        plt.ylabel(k)
                        plt.title('Barchart of ' + k)
                        plt.show()



    # Only histograms for numerical varibale
    button_hist = widgets.Button(description="Histogram")
    def hist(sender):
        suggestion3_func('hist')
    button_hist.on_click(hist)

    # Only boxplot for numerical variable
    button_boxp = widgets.Button(description="Box Plots")
    def boxplot1(sender):
        suggestion3_func('box_plot')
    button_boxp.on_click(boxplot1)

    # Only bar charts for categorical variable/

    button_bar = widgets.Button(description="Bar Graphs")
    def barplot(sender):
        suggestion3_func('bar')    
    button_bar.on_click(barplot)
    print()
    display(widgets.HBox((button_hist, button_boxp,button_bar)))


    print("===============Suggestion No. 4=============================")
    # Creating a folder in user specify directory 
       # Takin Input From The user
    text_direc = widgets.Text(placeholder="C:\\Users\\shubh\\Desktop\\MACHINE LEARNING-2",
                           description = "Directory Name : ",
                          style =style,disabled=False)
    text_fold = widgets.Text(placeholder="Eg.xyz",
                           description ="Folder Name : ",
                          style =style, disabled =False)
    print("Select the directory where you want to save your Plots.")
    display(text_direc)
    print("Name the folder in which you want to save your work.")
    display(text_fold)
    
    root = text_direc.value
    plot_dir = text_fold.value
    
    # Only histograms for numerical varibale
    button_h = widgets.Button(description="Histogram")
    def hist(sender):
        suggestion4_func('hist')
    button_h.on_click(hist)

    # Only boxplot for numerical variable
    button_b = widgets.Button(description="Box Plots")
    def boxplot1(sender):
        suggestion4_func('box_plot')
    button_b.on_click(boxplot1)

    # Only bar charts for categorical variable/
    button_ba = widgets.Button(description="Bar Graphs")
    def barplot(sender):
        suggestion4_func('bar')    
    button_ba.on_click(barplot)
    #print()
    # display(widgets.HBox((button_h, button_b, button_ba)))

    
    
       # defining the function for the folder create button
    def create(sender):
        if os.path.exists(text_direc.value):
            os.chdir(text_direc.value)
            if os.path.exists(text_direc.value + '\\' + text_fold.value) == False:
                os.mkdir(text_fold.value)
        else:
            print('Directory does not exist')
       # Creating a folder
    button_create = widgets.Button(description="Create Folder")
    print("Press the create button to create a folder in your given directory.")
    display(button_create)
    button_create.on_click(create)
    
    print("Press the button in which form you want to save your file : ")
    display(widgets.HBox((button_h, button_b, button_ba)))
    
    #Function to create and navigagte through folders for organising chart images
    def plot_directory(plot_value, root, plot_dir):
        #initially check if directory exists if not create..
        plot = '\\' + plot_dir
        plot_value = '\\' + plot_value
        if os.path.exists(root):
            if  os.getcwd() != root:
                os.chdir(root)
            if os.path.exists(root + plot) == False:
                os.mkdir(root + plot)
            plot_path = root + plot + plot_value
            if os.path.exists(plot_path) == False:
                os.mkdir(plot_path)
                os.chdir(plot_path)
            else:
                os.chdir(plot_path)
        else:
            print('Path does not exist')

#====================== Showing how to save the images into the folder ================= #
    #Create and save graphs in organised manner for user requested graphs and charts
    def suggestion4_func(value):
        df4 = load_file
        columns = df4.columns
        cat = []
        for t in columns:
            if((df4[t].dtype == np.float64) or (df4[t].dtype == np.int64)):
                if df4[t].nunique() <= 15:
                    cat.append('cat')
                else:
                    cat.append('num')
            else:
                cat.append('cat')
    # type your code here
        plot_directory(value, text_direc.value, text_fold.value)
        
        if value == 'hist':
            for h, k in enumerate(columns):
                if cat[h] == 'num':
                    df4.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
        elif value == 'box_plot':
            for h, k in enumerate(columns):
                if cat[h] == 'num':
                    df4.boxplot(column=k)
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')          
        elif value == 'bar':
            for h, k in enumerate(columns):
                if cat[h] == 'cat':
                    if df4[k].nunique() <= 15:
                        df4[k].value_counts().plot(kind = 'barh')
                        plt.xlabel('Count')
                        plt.ylabel(k)
                        plt.title('Histogram of ' + k)
                        plt.savefig(k + '.png')
     


# <h1 style="font-size:px; text-align:LEFT; margin-bottom:30px;">FORM 2: FUNCTION  </h1>

# In[9]:


def form2():
    # Step 1: In this step we are crating buttons to read the file.
    print("===============Exploratory Data Analysis ==================")
    print()
    print()
    print("===============Reading The Data Set =======================")
    style = {'description_width': '100px'}
    text_fd = widgets.Text(placeholder="C:\\Users\\shubh\\Desktop\\MACHINE LEARNING-2",
                           description = "Directory Name",
                          style =style,disabled=False)
    text_fn = widgets.Text(placeholder="Eg. \\xyz.csv",
                           description ="File Name",
                          style =style, disabled =False)
    print("Enter the directory where you data set is present in the format eg.C:\\Users\\shubh\\Desktop .")
    display(text_fd)
    print("Enter the name of the data set in the format eg. \\xyz.csv")
    display(text_fn)
    
    def readfile(sender):
        global load_file
        load_file = pd.read_csv( text_fd.value + '\\' + text_fn.value )
        print("Your File is loaded")

               #*****Button 1 - Reading the data set***** 
    button_read = widgets.Button(description="Read Data")
    button_read.on_click(readfile)  

    # ======================================================================================================================= #
    # Step 2:- In this step we giving some initial five rows.
    button_display = widgets.Button(description="Display Data")

    def displayfile(reader):
        display(load_file.head())

              #*****Button 2 - Displaying the data set***** 
    button_display.on_click(displayfile) 

    # ====================================================================================================================== #
    # Step 3:- In this step we are trying to summary of the data.
    def summary(file):
        display(load_file.describe())

               #*****Button 3 - Displaying summary of the data set***** 
    button_summary = widgets.Button(description="Summary")
    button_summary.on_click(summary)  

    # Displaying all the buttons in a single row
    display(widgets.HBox((button_read, button_display,button_summary))) # Displaying all buttons in a single row
    #======================================================MAIN==========================================================
    def graph_generator(graph_type, levels = 15):
        df = load_file
        #Splitting values taken from text box where column values were entered
        choice = text_specific.value.split(',')
        #if no value is specified then take all columns
        if choice[0] == '':
            choice[0] = 'all'
        columns = []
        var_type = []
        try:
            #Check if user entered column numbers or names and generate appropriate column names
            int(choice[0])
            for i,k in enumerate(choice):
                choice[i] = int(k)
            for i in choice:
                columns.append(df.columns[i])
        except:
            if (choice[0].lower() == 'all'):
                columns = df.columns
            elif type(choice[0]) == str:
                for i, k in enumerate(choice):
                    if k.strip() in df.columns:
                        columns.append(k.strip())
                    else:
                        print('Column does not exist')
                        return None
        #Find type of variable and append to a list...
        for t in columns:
            if((df[t].dtype == np.float64) or (df[t].dtype == np.int64)):
                #categorical values entered as numerical are entered here..
                if (df[t].nunique() <= levels):
                    var_type.append('cat')
                else:
                    var_type.append('num')
            else:
                var_type.append('cat')
        #If user specifies to output all graph types for particular or all columns..
        if graph_type == 'All':
            for h, k in enumerate(columns):
                if var_type[h] == 'num':
                    #histograms
                    value = 'histogram'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
                    plt.gcf().clear()
                    #boxplots
                    value = 'boxplot'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df.boxplot(column=k)
                    plt.ylabel('Count')
                    plt.title('Boxplot of ' + k)
                    plt.savefig(k + '.png') 
                    plt.gcf().clear()
                    #bar graphs
                elif (var_type[h] == 'cat') and (df[k].nunique() <= levels):
                    value = 'bar'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df[k].value_counts().plot(kind = 'barh')
                    plt.xlabel('Count')
                    plt.ylabel(k)
                    plt.title('Barchart of ' + k)
                    plt.savefig(k + '.png')
                    plt.gcf().clear()
                else:
                    print('Cannot print graphs for {}'.format(k))
        #If user specifies only histograms
        elif graph_type == 'histogram':
            for h, k in enumerate(columns):
                if var_type[h] == 'num':
                    #histograms
                    value = 'histogram'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df.hist(column=k)
                    plt.xlabel(k)
                    plt.ylabel('Count')
                    plt.title('Histogram of ' + k)
                    plt.savefig(k + '.png')
                    plt.gcf().clear()
                else:
                    print('Cant create Histogram for column {}'.format(k))
        #If user specifies only boxplots
        elif graph_type == 'boxplot':
            for h, k in enumerate(columns):
                if var_type[h] == 'num':
                    #boxplots
                    value = 'boxplot'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df.boxplot(column=k)
                    plt.ylabel('Count')
                    plt.title('Boxplot of ' + k)
                    plt.savefig(k + '.png')
                    plt.gcf().clear()
                else:
                    print('Cant create boxplot for column {}'.format(k))
        #If user specifies only bar charts
        elif graph_type == 'bar':
            for h, k in enumerate(columns):
                if (var_type[h] == 'cat') and (df[k].nunique() <= levels):
                    #histograms
                    value = 'bar'
                    plot_directory(value, text_direc.value, text_fold.value)
                    df[k].value_counts().plot(kind = 'barh')
                    plt.xlabel('Count')
                    plt.ylabel(k)
                    plt.title('Barchart of ' + k)
                    plt.savefig(k + '.png')
                    plt.gcf().clear()
                else:
                    print('Cant create bar chart for column {}'.format(k))   
    
    #function for creation of directories to save graphs
    def plot_directory(plot_value, root, plot_dir):
        #Initial check if path alraedy exists
        plot = '\\' + plot_dir
        plot_value = '\\' + plot_value
        if os.path.exists(root):
            if  os.getcwd() != root:
                os.chdir(root)
            if os.path.exists(root + plot) == False:
                os.mkdir(root + plot)
            plot_path = root + plot + plot_value
            if os.path.exists(plot_path) == False:
                os.mkdir(plot_path)
                os.chdir(plot_path)
            else:
                os.chdir(plot_path)
        else:
            print('Path does not exist')
        
    # ================================================================================================================ #
    # Creating a folder in user specify directory 
    # Takin Input From The user
    
    text_direc = widgets.Text(placeholder="C:\\Users\\shubh\\Desktop\\MACHINE LEARNING-2",
                           description = "Directory Name : ",
                          style =style,disabled=False)
    text_fold = widgets.Text(placeholder="Eg.xyz",
                           description ="Folder Name : ",
                          style =style, disabled =False)
    print("Select the directory where you want to save your Plots.")
    display(text_direc)
    print("Name the folder in which you want to save your work.")
    display(text_fold)
    
    # Only histograms for numerical varibale
    def hist_func(sender):
        graph_generator('histogram')
        
    button_h = widgets.Button(description="Histogram")
    button_h.on_click(hist_func)

    # Only boxplot for numerical variable
    def boxplot_func(sender):
        graph_generator('boxplot')
        
    button_b = widgets.Button(description="Box Plots")
    button_b.on_click(boxplot_func)

    # Only bar charts for categorical variable/
    def barplot_func(sender):
        graph_generator('bar')  
        
    button_ba = widgets.Button(description="Bar Graphs")
    button_ba.on_click(barplot_func)

      
        
    # defining the function for the folder create button
    def create(sender):
        if os.path.exists(text_direc.value):
            os.chdir(text_direc.value)
            if os.path.exists(text_direc.value + '\\' + text_fold.value) == False:
                os.mkdir(text_fold.value)
        else:
            print('Directory does not exist')
    
    print("Press the create button to create a folder in your given directory.")
    button_create = widgets.Button(description="Create Folder")
    display(button_create)
    button_create.on_click(create)
    print()
    print("===============Printing The Graphs =======================")
    
    
    #=================================================================================================================#
    # text box for entering columns    
    style = {'description_width': '250px'}
    text_specific = widgets.Text(placeholder="1,2,3",
                           description = "Give the coloumn name, no. or all :",
                          style =style,disabled=False)
    display(text_specific)
    
    
    #Button for all button
    def all_graph_generator(sender):
        graph_generator('All')
        
    button_all =widgets.Button(description="All")
    button_all.on_click(all_graph_generator)
    
    
    #==================================================================================================================#
    
    print("Press the button in which form you want to save your file : ")
    display(widgets.HBox((button_h, button_b, button_ba, button_all)))  
    
    print("=============== Measures of Central Tendency =======================")
    style = {'description_width': '250px'}
    text_mcd = widgets.Text(placeholder="1,2,3",
                           description ="Enter the column number : ",
                          style =style, disabled =False)
    print("Enter the coloumn which you want to calculate measure of central tendency.")
    display(text_mcd)
    #==========Calculation Of mean=========
    def mean(sender):
        choice = (text_mcd.value).split(',')
        a = load_file.columns
        df2 = load_file
        try:
            if choice[0] == 'all':
                for t in df2.columns:
                    if((df2[t].dtype == np.float64) or (df2[t].dtype == np.int64)):
                            print("The mean of the %s :"%t,load_file[t].mean())
            else:
                for i,k in enumerate(choice):
                    choice[i] = int(k)    
                for i in choice:
                    b=a[i]
                    print("The mean of the %s :"%b,df2[b].mean())
        except:
            print("Cannot calculate mean  of this coloumn because it is a categorical variable.")
    button_mcd =widgets.Button(description = "Mean")
    button_mcd.on_click(mean)
    #===============Calculation of median=====
    def median(sender):
        choice = (text_mcd.value).split(',')
        a = load_file.columns
        df2 = load_file
        try:
            if choice[0] == 'all':
                for t in df2.columns:
                    if((df2[t].dtype == np.float64) or (df2[t].dtype == np.int64)):
                            print("The median of the %s :"%t,load_file[t].median())
            else:
                for i,k in enumerate(choice):
                    choice[i] = int(k)    
                for i in choice:
                    b=a[i]
                    print("The median of the %s :"%b,df2[b].median())
        except:
            print("Cannot calculate median of this coloumn because it is a continous numerical variable.")
    button_med =widgets.Button(description = "Median")
    button_med.on_click(median)



    display(widgets.HBox((button_mcd, button_med)))
    
    print("=================== Obtaining the Scatter Matrix =======================")
    print("Press button to obtain Scatter Matrix.")
    def scatter_matrix(sender):
        pd.scatter_matrix(load_file,alpha = 0.3 ,figsize = (14,8), diagonal = 'kde')
        plt.show()
    button_scatter = widgets.Button(description = "Scatter Matrix")
    button_scatter.on_click(scatter_matrix)
    display(button_scatter)
    
    print("==================== Obtaining the Heatmap ============================")
    print()
    print("Press button to obtain the Heat Map.")
    def heat_map(sender):
        f,ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(load_file.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
        plt.show()

    button_heatmap = widgets.Button(description = "Heat Map")
    button_heatmap.on_click(heat_map)
    display(button_heatmap)

