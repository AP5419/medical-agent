# -*- coding: utf-8 -*-
"""初始数据库架构 — 创建全部 10 张核心业务表

Revision ID: 001
Revises:
Create Date: 2026-07-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """创建所有业务表。"""

    # ==================== 1. 科室表 ====================
    op.create_table(
        "departments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("name", sa.String(100), nullable=False, comment="科室名称"),
        sa.Column("description", sa.Text(), nullable=True, comment="科室描述"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        mysql_charset="utf8mb4",
    )

    # ==================== 2. 症状表 ====================
    op.create_table(
        "symptoms",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("name", sa.String(200), nullable=False, comment="症状名称"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_symptoms_name", "symptoms", ["name"])

    # ==================== 3. 疾病表 ====================
    op.create_table(
        "diseases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("name", sa.String(200), nullable=False, comment="疾病名称"),
        sa.Column("department_id", sa.BigInteger(), nullable=True, comment="所属科室ID"),
        sa.Column("description", sa.Text(), nullable=True, comment="疾病描述"),
        sa.Column("cause", sa.Text(), nullable=True, comment="病因"),
        sa.Column("prevent", sa.Text(), nullable=True, comment="预防措施"),
        sa.Column("cure_way", sa.Text(), nullable=True, comment="治疗方式"),
        sa.Column("cure_lasttime", sa.String(200), nullable=True, comment="治愈周期"),
        sa.Column("cured_prob", sa.String(200), nullable=True, comment="治愈概率"),
        sa.Column("cost_money", sa.String(200), nullable=True, comment="治疗费用"),
        sa.Column("easy_get", sa.Text(), nullable=True, comment="易感人群"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.ForeignKeyConstraint(["department_id"], ["departments.id"], ondelete="SET NULL"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_diseases_name", "diseases", ["name"])

    # ==================== 4. 疾病-症状关联表 ====================
    op.create_table(
        "disease_symptoms",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("disease_id", sa.BigInteger(), nullable=False, comment="疾病ID"),
        sa.Column("symptom_id", sa.BigInteger(), nullable=False, comment="症状ID"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["disease_id"], ["diseases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["symptom_id"], ["symptoms.id"], ondelete="CASCADE"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_ds_disease_id", "disease_symptoms", ["disease_id"])
    op.create_index("idx_ds_symptom_id", "disease_symptoms", ["symptom_id"])

    # ==================== 5. 药品表 ====================
    op.create_table(
        "drugs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("name", sa.String(200), nullable=False, comment="药品名称"),
        sa.Column("alias", sa.String(200), nullable=True, comment="药品别名"),
        sa.Column("category", sa.String(200), nullable=True, comment="药品类别"),
        sa.Column("manufacturer", sa.String(200), nullable=True, comment="生产厂家"),
        sa.Column("approval_number", sa.String(200), nullable=True, comment="批准文号"),
        sa.Column("is_otc", sa.Boolean(), nullable=True, comment="是否OTC"),
        sa.Column("stock_quantity", sa.Integer(), nullable=True, comment="库存数量"),
        sa.Column("price", sa.Float(), nullable=True, comment="单价"),
        sa.Column("expire_date", sa.String(50), nullable=True, comment="有效期"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_drugs_name", "drugs", ["name"])
    op.create_index("idx_drugs_category", "drugs", ["category"])

    # ==================== 6. 药品详情表 ====================
    op.create_table(
        "drug_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("drug_id", sa.BigInteger(), nullable=False, comment="药品ID"),
        sa.Column("indication", sa.Text(), nullable=True, comment="适应症"),
        sa.Column("usage_dosage", sa.Text(), nullable=True, comment="用法用量"),
        sa.Column("adverse_reaction", sa.Text(), nullable=True, comment="不良反应"),
        sa.Column("contraindication", sa.Text(), nullable=True, comment="禁忌"),
        sa.Column("precaution", sa.Text(), nullable=True, comment="注意事项"),
        sa.Column("interaction", sa.Text(), nullable=True, comment="药物相互作用"),
        sa.Column("storage", sa.Text(), nullable=True, comment="贮藏"),
        sa.Column("full_instruction", sa.Text(), nullable=True, comment="完整说明书"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("drug_id"),
        sa.ForeignKeyConstraint(["drug_id"], ["drugs.id"], ondelete="CASCADE"),
        mysql_charset="utf8mb4",
    )

    # ==================== 7. 疾病-药品关联表 ====================
    op.create_table(
        "disease_drugs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("disease_id", sa.BigInteger(), nullable=False, comment="疾病ID"),
        sa.Column("drug_id", sa.BigInteger(), nullable=False, comment="药品ID"),
        sa.Column("relation_type", sa.String(20), nullable=True, comment="关联类型(common/recommend)"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["disease_id"], ["diseases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["drug_id"], ["drugs.id"], ondelete="CASCADE"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_dd_disease_id", "disease_drugs", ["disease_id"])
    op.create_index("idx_dd_drug_id", "disease_drugs", ["drug_id"])

    # ==================== 8. 患者表 ====================
    op.create_table(
        "patients",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("name", sa.String(100), nullable=False, comment="患者姓名"),
        sa.Column("gender", sa.String(10), nullable=True, comment="性别"),
        sa.Column("age", sa.Integer(), nullable=True, comment="年龄"),
        sa.Column("phone", sa.String(50), nullable=True, comment="联系电话"),
        sa.Column("id_card", sa.String(50), nullable=True, comment="身份证号"),
        sa.Column("allergy_history", sa.Text(), nullable=True, comment="过敏史"),
        sa.Column("medical_history", sa.Text(), nullable=True, comment="既往病史"),
        sa.Column("blood_type", sa.String(10), nullable=True, comment="血型"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_patients_name", "patients", ["name"])

    # ==================== 9. 问诊记录表 ====================
    op.create_table(
        "consultations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("patient_id", sa.BigInteger(), nullable=False, comment="患者ID"),
        sa.Column("department_id", sa.BigInteger(), nullable=True, comment="科室ID"),
        sa.Column("chief_complaint", sa.Text(), nullable=True, comment="主诉"),
        sa.Column("diagnosis", sa.Text(), nullable=True, comment="诊断结果"),
        sa.Column("prescription", sa.Text(), nullable=True, comment="处方"),
        sa.Column("urgency_level", sa.String(20), nullable=False, server_default="normal", comment="紧急程度"),
        sa.Column("session_id", sa.String(100), nullable=True, comment="会话ID"),
        sa.Column("milvus_doc_id", sa.String(100), nullable=True, comment="Milvus文档ID"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["patient_id"], ["patients.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["department_id"], ["departments.id"], ondelete="SET NULL"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_consult_patient_id", "consultations", ["patient_id"])
    op.create_index("idx_consult_department_id", "consultations", ["department_id"])

    # ==================== 10. 用户表 ====================
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("username", sa.String(50), nullable=False, comment="用户名"),
        sa.Column("password_hash", sa.String(255), nullable=False, comment="密码哈希"),
        sa.Column("role", sa.String(20), nullable=False, server_default="patient", comment="用户角色"),
        sa.Column("real_name", sa.String(50), nullable=True, comment="真实姓名"),
        sa.Column("phone", sa.String(50), nullable=True, comment="联系电话"),
        sa.Column("email", sa.String(100), nullable=True, comment="邮箱"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true(), comment="是否激活"),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.false(), comment="是否已删除"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        mysql_charset="utf8mb4",
    )
    op.create_index("idx_users_username", "users", ["username"])
    op.create_index("idx_users_role", "users", ["role"])


def downgrade() -> None:
    """删除所有表（按外键依赖逆序删除）。"""
    op.drop_table("users")
    op.drop_table("consultations")
    op.drop_table("patients")
    op.drop_table("disease_drugs")
    op.drop_table("drug_details")
    op.drop_table("drugs")
    op.drop_table("disease_symptoms")
    op.drop_table("diseases")
    op.drop_table("symptoms")
    op.drop_table("departments")
