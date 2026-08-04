# -*- coding: utf-8 -*-
"""MySQL 数据初始化 — 导入科室、症状、药品、疾病、关联表及模拟患者"""
import json
import os
import re
import sys

import pymysql
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from medical_agent.core.config import get_settings

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "medical.json")


def load_jsonl(filepath: str) -> list[dict]:
    """加载 JSONL 格式的医疗数据文件"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_connection(settings) -> pymysql.Connection:
    """创建 MySQL 连接"""
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        charset="utf8mb4",
    )


def import_departments(conn, records: list[dict]) -> dict:
    """导入科室数据，返回 {名称: id} 映射"""
    dept_set = set()
    for rec in records:
        for cat in rec.get("category", []):
            if cat != "疾病百科":
                dept_set.add(cat)
        for dept in rec.get("cure_department", []):
            dept_set.add(dept)

    cursor = conn.cursor()
    dept_map = {}
    for name in tqdm(sorted(dept_set), desc="导入科室"):
        cursor.execute(
            "INSERT INTO departments (name) VALUES (%s) "
            "ON DUPLICATE KEY UPDATE id=id",
            (name,),
        )
        conn.commit()
        cursor.execute("SELECT id FROM departments WHERE name = %s", (name,))
        dept_map[name] = cursor.fetchone()[0]
    cursor.close()
    return dept_map


def import_symptoms(conn, records: list[dict]) -> dict:
    """导入症状数据，返回 {名称: id} 映射"""
    symptom_set = set()
    for rec in records:
        for s in rec.get("symptom", []):
            symptom_set.add(s.strip())

    cursor = conn.cursor()
    symptom_map = {}
    for name in tqdm(sorted(symptom_set), desc="导入症状"):
        cursor.execute(
            "INSERT INTO symptoms (name) VALUES (%s) "
            "ON DUPLICATE KEY UPDATE id=id",
            (name,),
        )
        conn.commit()
        cursor.execute("SELECT id FROM symptoms WHERE name = %s", (name,))
        symptom_map[name] = cursor.fetchone()[0]
    cursor.close()
    return symptom_map


def _parse_drug_detail(text: str):
    """解析 drug_detail 字符串，提取药品名称

    格式: "制造商药品名(别名)" 或 "药品名"
    返回 (full_name, drug_name_in_parens)
    """
    m = re.match(r"^(.+?)\((.+?)\)$", text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return text.strip(), None


def import_drugs(conn, records: list[dict]) -> dict:
    """导入药品数据，返回 {名称: id} 映射"""
    drug_map = {}  # name -> id

    cursor = conn.cursor()
    seen = set()

    for rec in tqdm(records, desc="导入药品"):
        # 从 common_drug 收集
        for name in rec.get("common_drug", []):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                cursor.execute(
                    "INSERT INTO drugs (name) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE id=id",
                    (name,),
                )

        # 从 recommand_drug 收集
        for name in rec.get("recommand_drug", []):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                cursor.execute(
                    "INSERT INTO drugs (name) VALUES (%s) "
                    "ON DUPLICATE KEY UPDATE id=id",
                    (name,),
                )

        # 从 drug_detail 收集（包含制造商信息）
        for detail_str in rec.get("drug_detail", []):
            prefix, drug_name = _parse_drug_detail(detail_str)
            target_name = drug_name or prefix
            if target_name and target_name not in seen:
                seen.add(target_name)
                if drug_name:
                    cursor.execute(
                        "INSERT INTO drugs (name, alias, manufacturer) VALUES (%s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE alias=VALUES(alias), manufacturer=VALUES(manufacturer)",
                        (drug_name, prefix, None),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO drugs (name) VALUES (%s) "
                        "ON DUPLICATE KEY UPDATE id=id",
                        (target_name,),
                    )

        conn.commit()

    # 查询所有药品 id
    cursor.execute("SELECT id, name FROM drugs")
    for row in cursor.fetchall():
        drug_map[row[1]] = row[0]

    cursor.close()
    return drug_map


def import_diseases(conn, records: list[dict], dept_map: dict) -> dict:
    """导入疾病数据，返回 {名称: id} 映射"""
    cursor = conn.cursor()

    # 预取所有科室category，建立快速查找: category -> 第一个非"疾病百科"的科室名
    disease_dept_map = {}
    for rec in records:
        name = rec.get("name", "")
        if not name:
            continue
        cats = [c for c in rec.get("category", []) if c != "疾病百科"]
        disease_dept_map[name] = cats[0] if cats else None

    disease_map = {}
    for rec in tqdm(records, desc="导入疾病"):
        name = rec.get("name", "")
        if not name:
            continue
        dept_name = disease_dept_map.get(name)
        dept_id = dept_map.get(dept_name) if dept_name else None

        cursor.execute(
            "INSERT INTO diseases (name, department_id, description, cause, prevent, "
            "cure_way, cure_lasttime, cured_prob, cost_money, easy_get) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "description=VALUES(description), cause=VALUES(cause), prevent=VALUES(prevent), "
            "cure_way=VALUES(cure_way), cure_lasttime=VALUES(cure_lasttime), "
            "cured_prob=VALUES(cured_prob), cost_money=VALUES(cost_money), easy_get=VALUES(easy_get)",
            (
                name,
                dept_id,
                rec.get("desc", ""),
                rec.get("cause", ""),
                rec.get("prevent", ""),
                json.dumps(rec.get("cure_way", []), ensure_ascii=False),
                rec.get("cure_lasttime", ""),
                rec.get("cured_prob", ""),
                rec.get("cost_money", ""),
                rec.get("easy_get", ""),
            ),
        )
        conn.commit()
        cursor.execute("SELECT id FROM diseases WHERE name = %s", (name,))
        disease_map[name] = cursor.fetchone()[0]

    cursor.close()
    return disease_map


def import_disease_symptoms(
    conn, records: list[dict], disease_map: dict, symptom_map: dict
) -> int:
    """导入疾病-症状关联"""
    cursor = conn.cursor()
    count = 0
    for rec in tqdm(records, desc="导入疾病-症状关联"):
        disease_name = rec.get("name", "")
        disease_id = disease_map.get(disease_name)
        if not disease_id:
            continue
        for sym_name in rec.get("symptom", []):
            sym_id = symptom_map.get(sym_name.strip())
            if not sym_id:
                continue
            cursor.execute(
                "INSERT INTO disease_symptoms (disease_id, symptom_id) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE id=id",
                (disease_id, sym_id),
            )
            count += 1
    conn.commit()
    cursor.close()
    return count


def import_disease_drugs(
    conn, records: list[dict], disease_map: dict, drug_map: dict
) -> int:
    """导入疾病-药品关联"""
    cursor = conn.cursor()
    count = 0
    for rec in tqdm(records, desc="导入疾病-药品关联"):
        disease_name = rec.get("name", "")
        disease_id = disease_map.get(disease_name)
        if not disease_id:
            continue

        # common_drug -> relation_type = "common"
        for drug_name in rec.get("common_drug", []):
            drug_id = drug_map.get(drug_name.strip())
            if not drug_id:
                continue
            cursor.execute(
                "INSERT INTO disease_drugs (disease_id, drug_id, relation_type) "
                "VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE id=id",
                (disease_id, drug_id, "common"),
            )
            count += 1

        # recommand_drug -> relation_type = "recommend"
        for drug_name in rec.get("recommand_drug", []):
            drug_id = drug_map.get(drug_name.strip())
            if not drug_id:
                continue
            cursor.execute(
                "INSERT INTO disease_drugs (disease_id, drug_id, relation_type) "
                "VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE id=id",
                (disease_id, drug_id, "recommend"),
            )
            count += 1

        # drug_detail -> relation_type = "recommend"
        for detail_str in rec.get("drug_detail", []):
            _, drug_name = _parse_drug_detail(detail_str)
            target = drug_name or detail_str.strip()
            drug_id = drug_map.get(target)
            if not drug_id:
                continue
            cursor.execute(
                "INSERT INTO disease_drugs (disease_id, drug_id, relation_type) "
                "VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE id=id",
                (disease_id, drug_id, "recommend"),
            )
            count += 1

    conn.commit()
    cursor.close()
    return count


def import_mock_patients(conn) -> int:
    """插入 3 条模拟患者记录"""
    cursor = conn.cursor()
    mock_patients = [
        ("张三", "男", 45, "O"),
        ("李四", "女", 58, "A"),
        ("王五", "男", 52, "B"),
    ]
    for name, gender, age, blood in mock_patients:
        cursor.execute(
            "INSERT INTO patients (name, gender, age, blood_type) VALUES (%s, %s, %s, %s)",
            (name, gender, age, blood),
        )
    conn.commit()
    cursor.close()
    return len(mock_patients)


def main():
    settings = get_settings()
    print(f"连接 MySQL: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")

    print(f"加载数据文件: {DATA_FILE}")
    records = load_jsonl(DATA_FILE)
    print(f"读取 {len(records)} 条疾病记录")

    conn = get_connection(settings)
    try:
        # [1] 科室
        print("\n[1/7] 导入科室数据...")
        dept_map = import_departments(conn, records)
        print(f"   >> 导入 {len(dept_map)} 个科室")

        # [2] 症状
        print("\n[2/7] 导入症状数据...")
        symptom_map = import_symptoms(conn, records)
        print(f"   >> 导入 {len(symptom_map)} 个症状")

        # [3] 药品
        print("\n[3/7] 导入药品数据...")
        drug_map = import_drugs(conn, records)
        print(f"   >> 导入 {len(drug_map)} 种药品")

        # [4] 疾病
        print("\n[4/7] 导入疾病数据...")
        disease_map = import_diseases(conn, records, dept_map)
        print(f"   >> 导入 {len(disease_map)} 种疾病")

        # [5] 疾病-症状关联
        print("\n[5/7] 导入疾病-症状关联...")
        ds_count = import_disease_symptoms(conn, records, disease_map, symptom_map)
        print(f"   >> 导入 {ds_count} 条关联")

        # [6] 疾病-药品关联
        print("\n[6/7] 导入疾病-药品关联...")
        dd_count = import_disease_drugs(conn, records, disease_map, drug_map)
        print(f"   >> 导入 {dd_count} 条关联")

        # [7] 模拟患者
        print("\n[7/7] 插入模拟患者数据...")
        patient_count = import_mock_patients(conn)
        print(f"   >> 插入 {patient_count} 条患者记录")

        # 汇总统计
        print("\n" + "=" * 50)
        print("MySQL 数据导入完成！汇总：")
        print(f"  科室:        {len(dept_map)}")
        print(f"  症状:        {len(symptom_map)}")
        print(f"  药品:        {len(drug_map)}")
        print(f"  疾病:        {len(disease_map)}")
        print(f"  疾病-症状:   {ds_count}")
        print(f"  疾病-药品:   {dd_count}")
        print(f"  患者:        {patient_count}")
        print("=" * 50)

    finally:
        conn.close()
        print("\nMySQL 连接已关闭")


if __name__ == "__main__":
    main()
