# -*- coding: utf-8 -*-
"""
Layer 6: 数据源层 - 仓储层
模块: 医疗数据通用 Repository
"""

from typing import Optional, Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.base_repository import BaseRepository
from medical_agent.models.medical import Department, Disease, Symptom, Drug, Patient, Consultation


class DepartmentRepository(BaseRepository[Department]):
    def __init__(self, db: AsyncSession):
        super().__init__(Department, db)

    async def get_by_name(self, name: str) -> Optional[Department]:
        return await self.find_one(name=name)


class DiseaseRepository(BaseRepository[Disease]):
    def __init__(self, db: AsyncSession):
        super().__init__(Disease, db)

    async def search_by_name(self, keyword: str, limit: int = 20) -> Sequence[Disease]:
        stmt = select(Disease).where(Disease.name.contains(keyword)).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()


class SymptomRepository(BaseRepository[Symptom]):
    def __init__(self, db: AsyncSession):
        super().__init__(Symptom, db)

    async def search_by_name(self, keyword: str, limit: int = 20) -> Sequence[Symptom]:
        stmt = select(Symptom).where(Symptom.name.contains(keyword)).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()


class DrugRepository(BaseRepository[Drug]):
    def __init__(self, db: AsyncSession):
        super().__init__(Drug, db)

    async def search_by_name(self, keyword: str, limit: int = 20) -> Sequence[Drug]:
        stmt = select(Drug).where(Drug.name.contains(keyword)).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()


class PatientRepository(BaseRepository[Patient]):
    def __init__(self, db: AsyncSession):
        super().__init__(Patient, db)

    async def get_by_name(self, name: str) -> Optional[Patient]:
        return await self.find_one(name=name)

    async def search_by_name(self, keyword: str, limit: int = 10) -> Sequence[Patient]:
        stmt = select(Patient).where(Patient.name.contains(keyword)).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()


class ConsultationRepository(BaseRepository[Consultation]):
    def __init__(self, db: AsyncSession):
        super().__init__(Consultation, db)

    async def get_by_patient(self, patient_id: int, limit: int = 20) -> Sequence[Consultation]:
        stmt = (
            select(Consultation)
            .where(Consultation.patient_id == patient_id)
            .order_by(Consultation.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_recent(self, limit: int = 50) -> Sequence[Consultation]:
        stmt = select(Consultation).order_by(Consultation.created_at.desc()).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def count_by_department(self) -> list[dict]:
        stmt = (
            select(Consultation.department_id, func.count(Consultation.id).label("cnt"))
            .group_by(Consultation.department_id)
        )
        result = await self.db.execute(stmt)
        return [{"department_id": r[0], "count": r[1]} for r in result.fetchall()]
