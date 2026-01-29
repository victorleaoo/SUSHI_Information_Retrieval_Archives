import json
import os

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FOLDER_METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'folders_metadata', 'old_FoldersV1.2.json')
SNC_TRANSLATION_PATH = './SncTranslationV1.3.xlsx'

class FolderLabelConstructor:
    def __init__(self):
        self.sncTranslation = {}

        with open(FOLDER_METADATA_PATH) as folderMetadataFile:
            self.folderMetadata = json.load(folderMetadataFile)
        
        self.sncExpansion = pd.ExcelFile(SNC_TRANSLATION_PATH).parse(pd.ExcelFile(SNC_TRANSLATION_PATH).sheet_names[0])

    # Create SNC Translation Dict
    def create_snc_translation(self):
        sncTranslation = {}
        sncList = self.sncExpansion['SNC'].dropna().tolist()
        for snc in sncList: # all snc codes
            label1965 = str(self.sncExpansion.loc[self.sncExpansion['SNC'] == snc, 1965].iloc[0])
            label1963 = str(self.sncExpansion.loc[self.sncExpansion['SNC'] == snc, 1963].iloc[0])

            # uses 1965 as main
            if label1965 != 'nan':
                label = label1965
            elif label1963 != 'nan':
                label = label1963
            else:
                print(f'Unable to translate {snc}')
                label = 'Unknown'
            
            raw_scope = str(self.sncExpansion.loc[self.sncExpansion['SNC'] == snc, 'Scope Note'].iloc[0])
            scope_truncated = str(self.sncExpansion.loc[self.sncExpansion['SNC'] == snc, 'Scope Note'].iloc[0])
            if raw_scope == 'nan':
                raw_scope = ''
                scope_truncated = ''
            if len(raw_scope)>0:
                stoppers = ['SEE', 'for which ', 'Exclude ', 'exclude ', 'Subdivide ', 'subdivide ', 'Other than ', 'other than ', 'For ', 'Except ', 'except ']
                cut = len(raw_scope)
                for stopper in stoppers:
                    newcut = raw_scope.find(stopper)
                    if newcut>=0 and newcut < cut: cut=newcut
                scope_truncated=raw_scope[:cut].strip()

            if '-' in snc:
                parent = snc.split('-')[0]
            elif ' ' in snc:
                parent = snc.split(' ')[0]
            else:
                parent = 'None'
            
            # Generate the translation for the snc code
            # Uses title, scope and parent (first element)
            sncTranslation[snc] = {}
            sncTranslation[snc]['label1965'] = label1965
            sncTranslation[snc]['label1963'] = label1963
            sncTranslation[snc]['raw_scope'] = raw_scope
            try:
                sncTranslation[snc]['scope_truncated'] = scope_truncated
            except:
                sncTranslation[snc]['scope_truncated'] = ''
            sncTranslation[snc]['main_title'] = label
            sncTranslation[snc]['parent'] = parent

        for snc in sncTranslation:
            parent_expanded = sncTranslation[snc]['main_title']
            current = snc
            while sncTranslation[current]['parent'] != 'None':
                current = sncTranslation[current]['parent']
                parent_expanded = sncTranslation[current]['main_title'] + ': ' + parent_expanded # add the titles of the parents: TITLE PARENT: TITLE EXPANSION
            sncTranslation[snc]['label_parent_expanded'] = parent_expanded

        return sncTranslation

    def create_full_snc_folder_label(self):
        """
        For each folder SNC, it gathers:
            - 1965 SNC description
            - 1963 SNC description
            - Raw Scope Note
            - Scope Note truncated by the stoppers
        """
        snc_lookup = self.create_snc_translation()
        updated_metadata = {}

        for folder_id, folder_data in self.folderMetadata.items():
            new_entry = folder_data.copy()
            folder_snc = folder_data.get('snc')

            new_entry.pop('folder_label', None)

            if folder_snc in snc_lookup:
                # Add the new values to the existing dictionary
                translation_info = snc_lookup[folder_snc]
                new_entry.update({
                    "label1965": translation_info['label1965'],
                    "label1963": translation_info['label1963'],
                    "raw_scope": translation_info['raw_scope'],
                    "scope_truncated": translation_info['scope_truncated'],
                    "main_title": translation_info['main_title'],
                    "parent": translation_info['parent'],
                    "label_parent_expanded": translation_info['label_parent_expanded']
                })
            else:
                new_entry['translation_status'] = "SNC not found in expansion table"

            updated_metadata[folder_id] = new_entry

        return updated_metadata

if __name__ == "__main__":
    folderlabel_c = FolderLabelConstructor()
    new_folder_metadata = folderlabel_c.create_full_snc_folder_label()
    with open("../data/folders_metadata/FoldersV1.3.json", 'w') as f:
        json.dump(new_folder_metadata, f, indent=4)