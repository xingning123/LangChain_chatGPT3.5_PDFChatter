import os

os.environ["OPENAI_API_KEY"] = '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'
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
from collections import OrderedDict
import json
import pandas as pd


def extract_headings(catalog):
    headings = []
    for i in range(len(catalog)):

        if not isinstance(catalog[i], list):

            title = catalog[i].title
            heading = {'title': title}

        else:

            heading['children'] = extract_headings(catalog[i])

        headings.append(heading)

    return headings


def remove_duplicates(lst):
    unique_dicts = list(OrderedDict.fromkeys(json.dumps(d, sort_keys=True) for d in lst))
    unique_lst = [json.loads(d) for d in unique_dicts]
    return unique_lst


def extract_titles(catalog):
    titles = []
    for item in catalog:
        if 'title' in item:
            L1 = item['title']
            if 'children' in item:
                for child in item['children']:
                    L2 = child['title']
                    if 'children' in child:
                        for grandchild in child['children']:
                            L3 = grandchild['title']
                            titles.append(L1 + '--' + L2 + '--' + L3)
                    else:
                        titles.append(L1 + '--' + L2)
        else:
            titles.append(L1)

    return titles


def check_dicts(DictOrList, keys):
    if not DictOrList:
        return False
    if isinstance(DictOrList, list):

        for d in DictOrList:
            if not all(key in d and d[key] for key in keys):
                return False

    else:

        if not all(key in DictOrList and DictOrList[key] for key in keys):
            return False

    return True


if __name__ == '__main__':

    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024)

    # 读取pdf文件
    loader = PyPDFLoader(r'02.  RESEAUX SOUPLES.pdf')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()

    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)

    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)

    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    docsearch = Chroma.from_documents(split_docs, embeddings)

    # 创建问答对象
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce",
                                     retriever=docsearch.as_retriever(),
                                     return_source_documents=True)

    oneshot_address = """

               {
                   "Road_number": "183",
                   "Road_name": "Avenue Valentin - Les Verres",
                   "Postal_code": "85160"
                   "City": "SAINT JEAN-DE-MONTS"
               }

               """

    # address
    prompt_template_for_address = f"""

    Your task is to find the construction address based on the content of this article.The construction address includes "Road_number", "Road_name", "Postal_code" and "City"
    Your task is to generate a JSON object with the keys "Road_number", "Road_name", "Postal_code" and "City". The values must be given in French.
    For example, the output is ：||| {oneshot_address} |||

    """

    # address_q = qa({"query": f"{prompt_template_for_address}"})
    address = {}
    while not address:
        address_q = qa({"query": f"{prompt_template_for_address}"})
        print(address_q['result'])
        key_address = ["Road_number", "Road_name", "Postal_code", "City"]
        try:
            new = json.loads(address_q['result'])
            if check_dicts(new, key_address):
                address = new
            print(address)
            series = pd.Series(address)
            df = pd.DataFrame(series).T
            df.to_csv('address.csv', sep=',', index=False, encoding='utf-8')
        except:
            print("no answer!")

    # lot
    oneshot_lot = """

               {
                   "Lot": "Lot 2 : RESEAUX SOUPLES",
               }

               """
    prompt_template_for_lot = f"""

        Your task is to find the "lot" based on the content of this article.
        Your task is to generate a JSON object with the keys "Lot". The values must be given in French.
        For example, the output is ：||| {oneshot_lot} |||

        """

    lot = {}
    while not lot:
        lot_q = qa({"query": f"{prompt_template_for_lot}"})
        print(lot_q['result'])
        key_lot = ["Lot"]
        try:
            new = json.loads(lot_q['result'])
            if check_dicts(new, key_lot) and len(new['Lot']) > 10:
                lot = new
            print(lot)
            with open('lot.json', 'w', encoding='utf-8') as file:
                json.dump(lot, file)
        except:
            print("no answer!")

    # 目录结构
    # 打开PDF文件
    with open(r'convert_02.  RESEAUX SOUPLES.pdf', 'rb') as file:
        # 创建PDF阅读器对象
        reader = PyPDF2.PdfFileReader(file)

        # 获取目录信息
        catalog = reader.getOutlines()
        # #目录输出
        # print(catalog)
        # #目录结构化输出
        # print(extract_headings(catalog[1:]))
        # 利用树状结构，到达每一个子标题。汇总每条链路，为之后的提问做准备
        Sommaire = remove_duplicates(extract_headings(catalog[1:]))

    oneshot_material = """

               {
               
                   "matériau_générique": BÉTON D’ENROBEMENT DES FOURREAUX,
                   "caractéristiques_techniques": Le béton d’enrobement des fourreaux sera dosé à 250 kg de ciment au mètre cube.
                   
               }

               """

    materials = []
    for i in extract_titles(Sommaire):

        prompt_template_for_Material = f"""
        As a document information extraction assistant, I will provide you with the appropriate headings so that you can easily find the content of the information under the heading.
        The materials include Building materials:Adhesives (e.g., resin, glue),Paint,Wallpaper,Plaster (e.g., cement, gypsum),Glass,Sand .etc and Interior decoration materials:Lighting fixtures,Curtains,Indoor furniture (e.g., chairs, tables, beds , cabinets).etc.
        The title is {i}
        Your task is to find the material and the corresponding use according to the content under the headings.
        Your task is to generate a JSON object with the keys "matériau_générique" and "caractéristiques_techniques". The values must be given in French.
        For example, the output is ：||| {oneshot_material} |||

        """

        material = {}
        material_q = qa({"query": f"{prompt_template_for_Material}"})
        print(material_q['result'])
        key_material = ["matériau_générique", "caractéristiques_techniques"]
        try:
            new = json.loads(material_q['result'])
            if check_dicts(new, key_material):
                L_lot = i.split('--')
                material['Sous_lot'] = L_lot[0]
                if len(L_lot) > 1:
                    material['Sous_sous_lot'] = L_lot[1]
                if len(L_lot) > 2:
                    material['Sous_sous_sous_lot'] = L_lot[2]
            if new['matériau_générique'] not in [d['matériau_générique'] for d in materials] and new[
                'caractéristiques_techniques'] not in [d['caractéristiques_techniques'] for d in materials]:
                material['matériau_générique'] = new['matériau_générique']
                material['caractéristiques_techniques'] = new['caractéristiques_techniques']

                print(material)
                materials.append(material)
            with open('material.json', 'w', encoding="utf-8") as file:
                json.dump(materials, file, ensure_ascii=False, indent=4, separators=(',', ': '))
            print("保存成功")
            print(len(materials))
        except:
            print("no answer!")

    # 创建DataFrame对象
    df = pd.DataFrame(materials)

    # 将DataFrame保存为CSV文件
    df.to_csv('material.csv', index=False, encoding='utf-8-sig')
