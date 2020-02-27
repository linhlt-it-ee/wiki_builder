from collections import defaultdict

import xlsxwriter as xlsxwriter
from xlsxwriter.worksheet import Worksheet
import utils.file_util as file_util

class Excel:
    def __init__(self, filename: str):
        self.workbook = xlsxwriter.Workbook(filename)

    def release(self):
        self.workbook.close()

    def get_sheet(self, name: str):
        worksheet = self.workbook.get_worksheet_by_name(name)
        if worksheet is None:
            worksheet = self.workbook.add_worksheet(name)

        return Sheet(worksheet)


class Sheet:
    def __init__(self, sheet: Worksheet):
        self.sheet = sheet
        self.indices = defaultdict(int)

    def add_cell(self, col_index: int, value: str):
        if self.indices[col_index] == 0:
            self.sheet.write_string(0, col_index, f'Level{col_index}')
            self.indices[col_index] += 1

        self.sheet.write_string(self.indices[col_index], col_index, value)
        self.indices[col_index] += 1


def demo(entity_level_dict):
    excel = Excel('entity_level_report.xlsx')
    for root_id in entity_level_dict:
        # print("Sheet name",root_id)
        root_sheet = excel.get_sheet(root_id)
        children = entity_level_dict[root_id]['children']
        root_sheet.add_cell(0, entity_level_dict[root_id]['label'])
        for child in children:
            child_id=child['id']
            child_level=child['level']
            child_label=child['label']
            child_short_names=child['short_names']
            child_description=child['description']
            path=child['path']
            child_content=child_id+'\n'+child_label+'\n'+child_description+'\n'+'|'.join(child_short_names)+'\n'+path
            root_sheet.add_cell(child_level, child_content)
        # root_sheet.add_cell(1, 'BBB')
        # root_sheet.add_cell(1, 'CCC')

    excel.release()


# if __name__ == '__main__':
#     demo(file_util.load_json("all_entity_level.json"))
