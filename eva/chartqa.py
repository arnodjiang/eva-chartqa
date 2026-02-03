import json,os
from loguru import logger


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """
    5 % numeric tolerance.  Exact match for non‑numeric.
    Implementation identical to pix2struct (avoids /0).
    """
    def _to_float(text: str):
        try:
            return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
        except ValueError:
            return None

    prediction, target = str(prediction).strip(), str(target).strip()
    p_float, t_float = _to_float(prediction), _to_float(target)

    # NB: the "and t_float" check is what prevents ZeroDivisionError
    if p_float is not None and t_float:
        rel_change = abs(p_float - t_float) / abs(t_float)
        return rel_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

def compute_accuracy(recs) -> float:
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)

class FileUtils:
    @staticmethod
    def filter_data(in_path, to_path, filter_id = "id"):
        """
        in_datas, pre_processed_datas, to_datas = FileUtils.filter_data(in_path, to_path, filter_id = "id")
        """
        pre_processed_datas = []
        processed_datas = []
        
        in_datas = FileUtils.load_file(in_path)
        logger.info(f"检查到输入文件有：{len(in_datas)}")
        
        if os.path.exists(to_path):
            to_datas = FileUtils.load_file(to_path)
            to_datas_ids = [i[filter_id] for i in to_datas]
            
            pre_processed_datas = [i for i in in_datas if i[filter_id] not in to_datas_ids]
            
            return in_datas, pre_processed_datas, to_datas
        else:
            return in_datas, in_datas, []
    @staticmethod
    def load_file(json_path):
        """
        datas = Utils.load_file(json_path)
        """
        if json_path.endswith(".jsonl"):
            return FileUtils._load_jsonl(json_path)
        elif json_path.endswith(".json"):
            return FileUtils._load_json(json_path)
        elif json_path.endswith(".parquet"):
            return FileUtils._load_parquet(json_path)
        else:
            raise FileNotFoundError(f"The file at '{json_path}' does not exist or is of an unsupported format.")

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
            
    @staticmethod
    def _load_jsonl(json_path):
        to_data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                itm_data = {
                    "imgname":    ex["meta_data"]["imgname"],
                    "query":      ex["meta_data"]["query"],
                    "prediction": ex["prediction"],
                    "answer":     ex["label"],
                    "split":      ex["meta_data"]["type"],
                }

                to_data.append(itm_data)
        return to_data

    @staticmethod
    def _load_parquet(parquet_path):
        to_data = []
        dataset = load_dataset("parquet", data_files=parquet_path)["train"] # dataset.num_rows
        for itm in dataset:
            to_data.append({
                "id": itm["imgname"]+"_"+itm["query"]+"_"+itm["type"],
                "file_path": Image.open(io.BytesIO(itm["image"])),
                "meta_data": {
                    "imgname": itm["imgname"],
                    "query": itm["query"],
                    "type": itm["type"]
                },
                "label": itm["label"]
            })
        return to_data
        
    @staticmethod
    def to_json(json_data, to_path: str):
        """
        将 Python 对象转换为 JSON 格式并写入指定文件。
        
        参数:
                json_data (Any): 要写入的 Python 对象（通常是字典或列表）
                to_path (str): 目标 JSON 文件的路径
        
        返回:
                None
        
        异常:
                PermissionError: 如果没有写入文件的权限
                TypeError: 如果对象包含无法序列化的类型
        
        使用示例:
                data = {"name": "John", "age": 30}
                JsonTool.to_json(data, "output.json")
        """
        # 确保目录存在
        directory = os.path.dirname(to_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(to_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    @staticmethod
    def to_jsonl(json_list, to_path: str):
        """
        将 Python 对象列表转换为 JSONL 格式并写入指定文件（每行一个对象）。
        
        参数:
                json_list (List[Any]): 要写入的 Python 对象列表（通常是字典的列表）
                to_path (str): 目标 JSONL 文件的路径
        
        返回:
                None
        
        异常:
                PermissionError: 如果没有写入文件的权限
                TypeError: 如果对象包含无法序列化的类型
        
        使用示例:
                data_list = [{"name": "John"}, {"name": "Jane"}]
                JsonTool.to_jsonl(data_list, "output.jsonl")
        """
        # 确保目录存在
        directory = os.path.dirname(to_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(to_path, "w", encoding="utf-8") as f:
            for item in json_list:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")

def main():
    file_path = ""
    datas = FileUtils.load_file(file_path)
    # evaluation
    preds_h = [i for i in datas if i["type"] == "human"]
    preds_a = [i for i in datas if i["type"] != "human"]
    acc_h = compute_accuracy(preds_h)
    acc_a = compute_accuracy(preds_a)
    total = len(preds_h) + len(preds_a)
    acc_o = (
        (acc_h * len(preds_h) + acc_a * len(preds_a)) / total
        if total else 0.0
    )

    # save evaluation
    eval_json = {
        "test_human":     round(acc_h * 100, 2),
        "test_augmented": round(acc_a * 100, 2),
        "overall":        round(acc_o * 100, 2),
    }

    # print summary
    print("\n────────  Finished inference  ────────")
    for k, v in eval_json.items():
        print(f"{k:>15}: {v:.2f}%")



if __name__ == "__main__":
    main()