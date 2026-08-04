# -*- coding: utf-8 -*-
"""Neo4j 知识图谱初始化 - 从 medical.json 构建医学知识图谱"""
import json
import os
import sys

from neo4j import GraphDatabase

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


def create_constraints(driver):
    """创建唯一性约束，确保节点不重复"""
    constraint_queries = [
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT department_name IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT check_name IF NOT EXISTS FOR (c:Check) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT food_name IF NOT EXISTS FOR (f:Food) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT producer_name IF NOT EXISTS FOR (p:Producer) REQUIRE p.name IS UNIQUE",
    ]
    with driver.session() as session:
        for query in constraint_queries:
            session.run(query)
        print("   >> 已创建 7 个唯一性约束")


def clear_database(driver):
    """清空所有节点和关系"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("   >> 已清空现有数据")


def import_data(driver, records: list[dict]):
    """批量导入疾病、症状、药品、科室等节点及关系"""
    # 节点数据容器
    disease_nodes = []       # {name, desc, cause, prevent, ...}
    symptom_names = set()    # 所有症状名称
    drug_names = set()       # 所有药品名称(通用名)
    department_names = set() # 所有科室名称
    check_names = set()      # 所有检查项名称
    food_do_eat = set()      # 宜吃食物
    food_no_eat = set()      # 忌吃食物
    producer_names = set()   # 生产厂家
    # 关系数据
    disease_symptom_rels = []    # (disease, symptom)
    disease_department_rels = [] # (disease, department)
    disease_common_drug = []     # (disease, drug)
    disease_recommend_drug = []  # (disease, drug)
    disease_check_rels = []      # (disease, check)
    disease_do_eat = []          # (disease, food)
    disease_no_eat = []          # (disease, food)
    disease_acompany = []        # (disease, disease)

    for rec in records:
        d_name = rec.get("name", "")
        if not d_name:
            continue

        # 收集科室
        for dept in rec.get("category", []):
            if dept != "疾病百科":
                department_names.add(dept)
                disease_department_rels.append((d_name, dept))
        for dept in rec.get("cure_department", []):
            department_names.add(dept)
            if (d_name, dept) not in disease_department_rels:
                disease_department_rels.append((d_name, dept))

        # 收集症状
        for sym in rec.get("symptom", []):
            symptom_names.add(sym)
            disease_symptom_rels.append((d_name, sym))

        # 收集药品
        for drug in rec.get("common_drug", []):
            drug_names.add(drug)
            disease_common_drug.append((d_name, drug))
        for drug in rec.get("recommand_drug", []):
            drug_names.add(drug)
            disease_recommend_drug.append((d_name, drug))

        # 收集检查项
        for check in rec.get("check", []):
            check_names.add(check)
            disease_check_rels.append((d_name, check))

        # 收集食物
        for food in rec.get("do_eat", []):
            food_do_eat.add(food)
            disease_do_eat.append((d_name, food))
        for food in rec.get("not_eat", []):
            food_no_eat.add(food)
            disease_no_eat.append((d_name, food))

        # 收集生产厂家（从 drug_detail 解析）
        for detail in rec.get("drug_detail", []):
            if "(" in detail:
                producer_name = detail[: detail.index("(")]
                if producer_name:
                    producer_names.add(producer_name)

        # 收集并发症（伴随疾病）
        for accompany in rec.get("acompany", []):
            disease_acompany.append((d_name, accompany))

        # 疾病节点属性
        disease_nodes.append({
            "name": d_name,
            "desc": rec.get("desc", ""),
            "cause": rec.get("cause", ""),
            "prevent": rec.get("prevent", ""),
            "cure_way": "; ".join(rec.get("cure_way", [])),
            "cure_lasttime": rec.get("cure_lasttime", ""),
            "cured_prob": rec.get("cured_prob", ""),
            "cost_money": rec.get("cost_money", ""),
            "easy_get": rec.get("easy_get", ""),
            "get_prob": rec.get("get_prob", ""),
            "get_way": rec.get("get_way", ""),
        })

    with driver.session() as session:
        # 批量创建 Symptom 节点
        print(f"   >> 创建 {len(symptom_names)} 个症状节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Symptom {name: item})",
            items=list(symptom_names),
        )

        # 批量创建 Drug 节点
        print(f"   >> 创建 {len(drug_names)} 个药品节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Drug {name: item})",
            items=list(drug_names),
        )

        # 批量创建 Department 节点
        print(f"   >> 创建 {len(department_names)} 个科室节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Department {name: item})",
            items=list(department_names),
        )

        # 批量创建 Check 节点
        print(f"   >> 创建 {len(check_names)} 个检查项节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Check {name: item})",
            items=list(check_names),
        )

        # 批量创建 Food 节点
        all_foods = food_do_eat | food_no_eat
        print(f"   >> 创建 {len(all_foods)} 个食物节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Food {name: item})",
            items=list(all_foods),
        )

        # 批量创建 Producer 节点
        print(f"   >> 创建 {len(producer_names)} 个生产厂家节点...")
        session.run(
            "UNWIND $items AS item MERGE (n:Producer {name: item})",
            items=list(producer_names),
        )

        # 批量创建 Disease 节点（带属性）
        print(f"   >> 创建 {len(disease_nodes)} 个疾病节点（含属性）...")
        session.run(
            """
            UNWIND $items AS item
            MERGE (n:Disease {name: item.name})
            SET n.description = item.desc,
                n.cause = item.cause,
                n.prevent = item.prevent,
                n.cure_way = item.cure_way,
                n.cure_lasttime = item.cure_lasttime,
                n.cured_prob = item.cured_prob,
                n.cost_money = item.cost_money,
                n.easy_get = item.easy_get,
                n.get_prob = item.get_prob,
                n.get_way = item.get_way
            """,
            items=disease_nodes,
        )

        # 创建关系：HAS_SYMPTOM
        print(f"   >> 创建 {len(disease_symptom_rels)} 条 HAS_SYMPTOM 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (s:Symptom {name: rel[1]})
            MERGE (d)-[:HAS_SYMPTOM]->(s)
            """,
            rels=disease_symptom_rels,
        )

        # 创建关系：BELONGS_TO
        print(f"   >> 创建 {len(disease_department_rels)} 条 BELONGS_TO 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (dept:Department {name: rel[1]})
            MERGE (d)-[:BELONGS_TO]->(dept)
            """,
            rels=disease_department_rels,
        )

        # 创建关系：COMMON_DRUG
        print(f"   >> 创建 {len(disease_common_drug)} 条 COMMON_DRUG 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (dr:Drug {name: rel[1]})
            MERGE (d)-[:COMMON_DRUG]->(dr)
            """,
            rels=disease_common_drug,
        )

        # 创建关系：RECOMMEND_DRUG
        print(f"   >> 创建 {len(disease_recommend_drug)} 条 RECOMMEND_DRUG 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (dr:Drug {name: rel[1]})
            MERGE (d)-[:RECOMMEND_DRUG]->(dr)
            """,
            rels=disease_recommend_drug,
        )

        # 创建关系：NEED_CHECK
        print(f"   >> 创建 {len(disease_check_rels)} 条 NEED_CHECK 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (c:Check {name: rel[1]})
            MERGE (d)-[:NEED_CHECK]->(c)
            """,
            rels=disease_check_rels,
        )

        # 创建关系：DO_EAT
        print(f"   >> 创建 {len(disease_do_eat)} 条 DO_EAT 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (f:Food {name: rel[1]})
            MERGE (d)-[:DO_EAT]->(f)
            """,
            rels=disease_do_eat,
        )

        # 创建关系：NO_EAT
        print(f"   >> 创建 {len(disease_no_eat)} 条 NO_EAT 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d:Disease {name: rel[0]}), (f:Food {name: rel[1]})
            MERGE (d)-[:NO_EAT]->(f)
            """,
            rels=disease_no_eat,
        )

        # 创建关系：ACOMPANY_WITH
        print(f"   >> 创建 {len(disease_acompany)} 条 ACOMPANY_WITH 关系...")
        session.run(
            """
            UNWIND $rels AS rel
            MATCH (d1:Disease {name: rel[0]}), (d2:Disease {name: rel[1]})
            MERGE (d1)-[:ACOMPANY_WITH]->(d2)
            """,
            rels=disease_acompany,
        )

    # 返回统计
    return {
        "diseases": len(disease_nodes),
        "symptoms": len(symptom_names),
        "drugs": len(drug_names),
        "departments": len(department_names),
        "checks": len(check_names),
        "foods": len(all_foods),
        "producers": len(producer_names),
        "relations": (
            len(disease_symptom_rels)
            + len(disease_department_rels)
            + len(disease_common_drug)
            + len(disease_recommend_drug)
            + len(disease_check_rels)
            + len(disease_do_eat)
            + len(disease_no_eat)
            + len(disease_acompany)
        ),
    }


def main():
    """主入口：清空并重建 Neo4j 知识图谱"""
    settings = get_settings()
    print(f"连接 Neo4j: {settings.NEO4J_URI}")

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()
        print("   >> Neo4j 连接成功")

        # 清空现有数据
        print("\n[1/4] 清空现有数据...")
        clear_database(driver)

        # 创建约束
        print("\n[2/4] 创建唯一性约束...")
        create_constraints(driver)

        # 加载数据
        print(f"\n[3/4] 加载数据文件: {DATA_FILE}")
        records = load_jsonl(DATA_FILE)
        print(f"   >> 读取 {len(records)} 条疾病记录")

        # 导入数据
        print("\n[4/4] 导入节点和关系...")
        stats = import_data(driver, records)

        # 汇总统计
        print("\n" + "=" * 50)
        print("Neo4j 知识图谱构建完成！汇总：")
        print(f"  疾病节点: {stats['diseases']}")
        print(f"  症状节点: {stats['symptoms']}")
        print(f"  药品节点: {stats['drugs']}")
        print(f"  科室节点: {stats['departments']}")
        print(f"  检查项节点: {stats['checks']}")
        print(f"  食物节点: {stats['foods']}")
        print(f"  生产厂家节点: {stats['producers']}")
        print(f"  关系总数: {stats['relations']}")
        print("=" * 50)

    finally:
        driver.close()
        print("\nNeo4j 连接已关闭")


if __name__ == "__main__":
    main()
