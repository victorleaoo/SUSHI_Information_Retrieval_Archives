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