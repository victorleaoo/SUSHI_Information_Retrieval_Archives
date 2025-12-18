# Functions that are not strictly relevant for the model creation and run

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt

#################################
# Count of folders for each SNC #
#################################
def count_snc(folders):
    counts = {}
    for folder in folders:
        if 'snc' in folders[folder]:
            snc = folders[folder]['snc']
            if snc not in counts:
                counts[snc]=0
            counts[snc]+=1
    counts = dict(sorted(counts.items()))
    print(counts)

#############################################
# PLOTTING COSSINE SIMILARITY BETWEEN BOXES #
#############################################
def create_heatmap(similarity, labels1, labels2, cmap = "YlGnBu"):
  df = pd.DataFrame(similarity)
  df.columns = labels1
  df.index = labels2
  fig, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(df, cmap=cmap)

def load_box(items, box):
    titles=[]
    docs=[]
    for i, item in enumerate(items):
        if items[item]['Sushi Box']==box:
            ocrtext=''
            for j, text in enumerate(items[item]['ocr']):
                ocrtext+=text
                # print(f'Processed document {i} OCR page {j}')
            docs.append(text)
            titles.append(items[item]['title'])
            # boxes.append(items[item]['Sushi Box'])
            # print(titles[i])
    return titles, docs

def load_collection(items):
    titles=[]
    docs=[]
    for i, item in enumerate(items):
        ocrtext=''
        for j, text in enumerate(items[item]['ocr']):
            if j==0: ocrtext+=text
            # print(f'Processed document {i} OCR page {j}')
        #docs.append(text)
        docs.append(items[item]['summary'])
        titles.append(items[item]['title'])
    return titles, docs

def checksim(items, box1, box2):
    print('Starting checksim')

    item_array = items.items()
    box_sorted_item_array = sorted(item_array, key=lambda item: item[1]['Sushi Box'])
    start = {}
    end = {}
    prior_box = ''
    for i, doc in enumerate(box_sorted_item_array):
        if prior_box == '':
            start[box_sorted_item_array[i][1]['Sushi Box']] = i
            prior_box = doc[1]['Sushi Box']
        elif doc[1]['Sushi Box'] != prior_box:
            end[prior_box] = i
            box_sorted_item_array = box_sorted_item_array[:start[prior_box]]+sorted(box_sorted_item_array[start[prior_box]:end[prior_box]], key=lambda item: item[1]['Sushi Folder'])+box_sorted_item_array[end[prior_box]:]
            prior_box = doc[1]['Sushi Box']
            start[prior_box] = i
    end[prior_box] = i+1

    # Index when each box start
    #print(f'Start: {start}')
    # Index when each box ends
    #print(f'End: {end}')

    items = dict(box_sorted_item_array[17652:])
    #titles, docs = load_box(items, box1)
    titles, docs = load_collection(items)
    #box1_size=len(titles)
    #titles2, docs2 = load_box(items, box2)
    #titles += titles2
    #docs += docs2
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    arr = X.toarray()
    sim_array=cosine_similarity(arr)
    #sum=0
    #for i, row in enumerate(sim_array[box1_size:]):
    #    for column in row[box1_size:]:
    #        sum+=column
    #elements = len(sim_array)*len(sim_array[0])/2-len(sim_array)
    #print(f'Sum: {sum}, Elements: {elements}. Average Similarity: {sum/elements:.3f}')
    #for i, row in enumerate(sim_array[:box1_size]):
    #    for column in row[:box1_size]:
    #        sum+=column
    #elements = len(sim_array)*len(sim_array[0])/2-len(sim_array)
    #print(f'Sum: {sum}, Elements: {elements}. Average Similarity: {sum/elements:.3f}')
    #for i, row in enumerate(sim_array[box1_size:]):
    #    for column in row[:box1_size]:
    #        sum+=column
    #elements = len(sim_array)*len(sim_array[0])/2-len(sim_array)
    #print(f'Sum: {sum}, Elements: {elements}. Average Similarity: {sum/elements:.3f}')

    #print(sim_array)
    create_heatmap(sim_array, titles, titles)
    plt.show()

##############################
# Plot Folders Dates Density #
##############################
def plot_dates(title, dates, pdf, cutoff):
    #print(f'Type of dates: {type(dates)}')
    #print(dates)
    df = pd.DataFrame(dates)
    df = df[df['date']!='Unknown']
    # print(df)
    # Convert dates to the numeric format for plotting
    df['Date_ordinal'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").toordinal())
    df = df.head(cutoff)
    if df.shape[0]>1:
        # Create a density plot
        sns.kdeplot(df['Date_ordinal'])
        plt.xlabel('Date')
        plt.ylabel('Density')
        plt.title(title+' ('+str(df.shape[0])+')')
        # Format the x-axis to show dates instead of ordinal numbers
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: date.fromordinal(int(x)).strftime('%Y-%m-%d')))
        plt.xlim(pd.to_datetime('1962-01-01').toordinal(), pd.to_datetime('1977-01-01').toordinal())
        pdf.savefig()
        plt.clf()
        # plt.show()

def plotFolderDates(folderMetadata):
    dates=[]
    pdf = PdfPages('./dates/folderplot.pdf')
    #print(f'Folders with unknown dates')
    for folder in folderMetadata:
        if folderMetadata[folder]['date']=='Unknown':
            x=2
            #print(f'Folder {folder}   SNC: {folderMetadata[folder]["snc"]:13} Label: {folderMetadata[folder]["label"]}  ')
        else:
            if folderMetadata[folder]['date'] == '1964': print(f'1964 in folder {folder} with label {folderMetadata[folder]["label"]}')
            dates.append(folderMetadata[folder]['date'])
    #print(dates)
    df = pd.DataFrame(dates, columns=['original_date'])
    df['date'] = pd.to_datetime(df['original_date'], format='mixed')
    df['date'] = df['date'].dt.strftime('%y-%m-%d')
    df['date'] = df['date'].apply(lambda x: '19'+x)
    dates = df.iloc[:, 1]
    plot_dates('Folders', dates, pdf, 1000)
    pdf.close()
    f=open('./dates/folderdates.txt', 'w')
    datelist = dates.tolist()
    for datestring in datelist:
        print(datestring, file=f)
    f.close()