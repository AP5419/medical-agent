# -*- coding: utf-8 -*-
# 医疗核心数据模型 - 科室、疾病、症状、药品、患者、问诊等 ORM 模型
# 企业架构说明: MySQL 代表医院 HIS 系统(权威数据源)，Neo4j 为只读查询副本。
# 药品/科室等运营数据以 HIS MySQL 为准，通过 ETL 管道同步到 Neo4j。
# 疾病-症状-药品等关系数据以 medical.json 为准，同源初始化到 MySQL 和 Neo4j。

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from medical_agent.core.base_model import BaseModel


class Department(BaseModel):
    """科室模型 - HIS 权威数据源"""

    __tablename__ = "departments"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="科室名称")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="科室描述")


class Disease(BaseModel):
    """疾病模型 - medical.json 初始化，手动维护"""

    __tablename__ = "diseases"

    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="疾病名称")
    department_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True, comment="所属科室ID"
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="疾病描述")
    cause: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="病因")
    prevent: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="预防措施")
    cure_way: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="治疗方式")
    cure_lasttime: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="治愈周期")
    cured_prob: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="治愈概率")
    cost_money: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="治疗费用")
    easy_get: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="易感人群")

    __table_args__ = (Index("idx_diseases_name", "name"),)
    department: Mapped[Optional["Department"]] = relationship("Department", backref="diseases")


class Symptom(BaseModel):
    """症状模型 - medical.json 初始化"""

    __tablename__ = "symptoms"

    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, comment="症状名称")
    __table_args__ = (Index("idx_symptoms_name", "name"),)


class DiseaseSymptom(BaseModel):
    """疾病-症状关联（多对多中间表）"""

    __tablename__ = "disease_symptoms"

    disease_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False, comment="疾病ID"
    )
    symptom_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symptoms.id", ondelete="CASCADE"), nullable=False, comment="症状ID"
    )
    __table_args__ = (Index("idx_ds_disease_id", "disease_id"), Index("idx_ds_symptom_id", "symptom_id"))
    disease: Mapped["Disease"] = relationship("Disease", backref="disease_symptoms")
    symptom: Mapped["Symptom"] = relationship("Symptom", backref="disease_symptoms")


class Drug(BaseModel):
    """药品模型 - HIS 权威数据源，药剂科维护"""

    __tablename__ = "drugs"

    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="药品名称")
    alias: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="药品别名")
    category: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="药品类别")
    manufacturer: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="生产厂家")
    approval_number: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, comment="批准文号")
    is_otc: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, comment="是否OTC")
    stock_quantity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="库存数量")
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="单价")
    expire_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="有效期")

    __table_args__ = (Index("idx_drugs_name", "name"), Index("idx_drugs_category", "category"))


class DrugDetail(BaseModel):
    """药品详情（一对一关联药品）"""

    __tablename__ = "drug_details"

    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drugs.id", ondelete="CASCADE"), unique=True, nullable=False, comment="药品ID"
    )
    indication: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="适应症")
    usage_dosage: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="用法用量")
    adverse_reaction: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="不良反应")
    contraindication: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="禁忌")
    precaution: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="注意事项")
    interaction: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="药物相互作用")
    storage: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="贮藏")
    full_instruction: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="完整说明书")
    drug: Mapped["Drug"] = relationship("Drug", backref="detail")


class DiseaseDrug(BaseModel):
    """疾病-药品关联（多对多中间表）"""

    __tablename__ = "disease_drugs"

    disease_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("diseases.id", ondelete="CASCADE"), nullable=False, comment="疾病ID"
    )
    drug_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drugs.id", ondelete="CASCADE"), nullable=False, comment="药品ID"
    )
    relation_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, comment="关联类型(common/recommend)"
    )
    __table_args__ = (Index("idx_dd_disease_id", "disease_id"), Index("idx_dd_drug_id", "drug_id"))
    disease: Mapped["Disease"] = relationship("Disease", backref="disease_drugs")
    drug: Mapped["Drug"] = relationship("Drug", backref="disease_drugs")


class Patient(BaseModel):
    """患者模型 - HIS 权威数据源"""

    __tablename__ = "patients"

    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="患者姓名")
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="性别")
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="年龄")
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="联系电话")
    id_card: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="身份证号")
    allergy_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="过敏史")
    medical_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="既往病史")
    blood_type: Mapped[Optional[str]] = mapped_column(String(10), nullable=True, comment="血型")

    __table_args__ = (Index("idx_patients_name", "name"),)


class Consultation(BaseModel):
    """问诊记录模型 - HIS 权威数据源"""

    __tablename__ = "consultations"

    patient_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, comment="患者ID"
    )
    department_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True, comment="科室ID"
    )
    chief_complaint: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="主诉")
    diagnosis: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="诊断结果")
    prescription: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="处方")
    urgency_level: Mapped[str] = mapped_column(
        String(20), nullable=False, default="normal", comment="紧急程度"
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="会话ID")
    milvus_doc_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="Milvus文档ID")

    __table_args__ = (
        Index("idx_consult_patient_id", "patient_id"),
        Index("idx_consult_department_id", "department_id"),
    )
    patient: Mapped["Patient"] = relationship("Patient", backref="consultations")
    department: Mapped[Optional["Department"]] = relationship("Department", backref="consultations")
