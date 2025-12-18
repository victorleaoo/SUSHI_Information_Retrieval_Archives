def translate_snc_label(naraLabel, brownLabel, sushiFile, folder, sncExpansion):
    if not isinstance(naraLabel, float):
        start = naraLabel.find('(') # Strip part markings
        if start != -1:
            naraLabel = naraLabel[:start]
        naraLabel = naraLabel.replace('BRAZ-A0', 'BRAZ-A 0') # Fix formatting error
        naraLabel = naraLabel.replace('BRAZ-E0', 'BRAZ-E 0')  # Fix formatting error
        naraLabelElements = naraLabel.split()
        if len(naraLabelElements) in [3,4]:
            if len(naraLabelElements)==3:
                naraSnc = naraLabelElements[0]
            else:
                naraSnc = ' '.join(naraLabelElements[0:2])
            # naraCountryCode = naraLabelElements[-2]
            # naraDate = naraLabelElements[-1]
            #print(f'parsed {naraLabel} to {naraSnc} // {naraCountryCode} // {naraDate}')
            if naraSnc in sncExpansion['SNC'].tolist():
                label1965 = str(sncExpansion.loc[sncExpansion['SNC']==naraSnc, 1965].iloc[0])
                label1963 = str(sncExpansion.loc[sncExpansion['SNC']==naraSnc, 1963].iloc[0])
                if label1965 != 'nan':
                    label = label1965
                elif label1963 != 'nan':
                    label = label1963
                else:
                    #print(f'Unable to translate {naraSnc} for file {sushiFile} in folder {folder}')
                    label=naraSnc
            else:
                #print(f'No expansion for {naraSnc}')
                label = naraSnc

        else:
            #print(f"NARA Folder Title doesn't have four parts: {naraLabel}")
            label = 'Bad NARA Folder Title'
    else:
        if not isinstance(brownLabel, float):
            label = brownLabel.replace('_', ' ')
        else:
            #print(f'Missing both NARA and Brown folder labels for file {sushiFile} in folder {folder}')
            label = 'No NARA or Brown Folder Title'

    return label

import pandas as pd

def load_snc_expansions(snc_translation_path):
    print(f'Loading SNC expansion table')
    try:
        xls = pd.ExcelFile(snc_translation_path)
        sncExpansion = xls.parse(xls.sheet_names[0])
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(-1)

    sncTranslation = {}
    sncList = sncExpansion['SNC'].dropna().tolist()
    for snc in sncList: # all snc codes
        label1965 = str(sncExpansion.loc[sncExpansion['SNC'] == snc, 1965].iloc[0])
        label1963 = str(sncExpansion.loc[sncExpansion['SNC'] == snc, 1963].iloc[0])

        # uses 1965 as main
        if label1965 != 'nan':
            label = label1965
        elif label1963 != 'nan':
            label = label1963
        else:
            print(f'Unable to translate {snc}')
            label = 'Unknown'
        
        scope = str(sncExpansion.loc[sncExpansion['SNC'] == snc, 'Scope Note'].iloc[0])
        if scope == 'nan':
            scope = ''
        if len(scope)>0:
            stoppers = ['SEE', 'for which ', 'Exclude ', 'exclude ', 'Subdivide ', 'subdivide ', 'Other than ', 'other than ', 'For ', 'Except ', 'except ']
            cut = len(scope)
            for stopper in stoppers:
                newcut = scope.find(stopper)
                if newcut>=0 and newcut < cut: cut=newcut
            scope=scope[:cut]

        if '-' in snc:
            parent = snc.split('-')[0]
        elif ' ' in snc:
            parent = snc.split(' ')[0]
        else:
            parent = 'None'
        
        # Generate the translation for the snc code
        # Uses title, scope and parent (first element)
        sncTranslation[snc] = {}
        sncTranslation[snc]['title'] = label
        sncTranslation[snc]['scope'] = scope
        sncTranslation[snc]['parent'] = parent

    for snc in sncTranslation:
        expanded = sncTranslation[snc]['title']
        current = snc
        while sncTranslation[current]['parent'] != 'None':
            current = sncTranslation[current]['parent']
            expanded = sncTranslation[current]['title'] + ': ' + expanded # add the titles of the parents: TITLE PARENT: TITLE EXPANSION
        sncTranslation[snc]['expanded'] = expanded

    #for snc in sncTranslation:
    #    print(f'SNC {snc:10} has translation {sncTranslation[snc]}')
    #print(f'Read {len(sncTranslation)} SNC expansions')
    return sncTranslation