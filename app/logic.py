from __future__ import annotations
from typing import List
import re

from .models import Roadmap, Milestone, Resource


BASIC_MILESTONES = [
    Milestone(
        id='sql-basics',
        title='Основы SQL и селекты',
        summary='SELECT, FROM, WHERE, базовые типы данных, NULL.',
        topics=['SELECT', 'FROM', 'WHERE', 'Типы данных', 'NULL', 'Операторы сравнения'],
        resources=[
            Resource(
                title='Mode: SQL Tutorial', url='https://mode.com/sql-tutorial/', type='course'
            )
        ],
        estimated_hours=6,
        tags=['core'],
    ),
    Milestone(
        id='joins-agg',
        title='JOIN и агрегации',
        summary='INNER/LEFT/RIGHT/FULL JOIN, GROUP BY, HAVING.',
        topics=['JOIN', 'GROUP BY', 'HAVING', 'DISTINCT'],
        resources=[
            Resource(
                title='PostgreSQL docs: SELECT',
                url='https://www.postgresql.org/docs/current/queries-select.html',
                type='doc',
            )
        ],
        estimated_hours=6,
        tags=['core'],
    ),
    Milestone(
        id='subqueries-ctes',
        title='Подзапросы и CTE',
        summary='Подзапросы в SELECT/WHERE, WITH/CTE, рекурсивные CTE.',
        topics=['Subquery', 'CTE', 'WITH RECURSIVE'],
        resources=[
            Resource(
                title='CTE Explained',
                url='https://www.sqltutorial.org/sql-cte/',
                type='article',
            )
        ],
        estimated_hours=5,
        tags=['core'],
    ),
    Milestone(
        id='indexes-perf',
        title='Индексы и производительность',
        summary='Индексы, планы выполнения, оптимизация запросов.',
        topics=['Indexes', 'EXPLAIN/EXPLAIN ANALYZE', 'Query Tuning'],
        resources=[
            Resource(
                title='Use The Index, Luke!', url='https://use-the-index-luke.com/', type='book'
            )
        ],
        estimated_hours=6,
        tags=['perf'],
    ),
    Milestone(
        id='transactions',
        title='Транзакции и блокировки',
        summary='ACID, уровни изоляции, блокировки, deadlocks.',
        topics=['ACID', 'Isolation Levels', 'Locks', 'Deadlocks'],
        resources=[
            Resource(
                title='PostgreSQL Concurrency Control',
                url='https://www.postgresql.org/docs/current/mvcc.html',
                type='doc',
            )
        ],
        estimated_hours=4,
        tags=['db-theory'],
    ),
    Milestone(
        id='window-funcs',
        title='Оконные функции',
        summary='OVER, PARTITION BY, ORDER BY, ROW_NUMBER, LAG/LEAD.',
        topics=['Window Functions', 'OVER()', 'PARTITION BY', 'LAG/LEAD'],
        resources=[
            Resource(
                title='PostgreSQL: Window Functions',
                url='https://www.postgresql.org/docs/current/functions-window.html',
                type='doc',
            )
        ],
        estimated_hours=5,
        tags=['analytics'],
    ),
    Milestone(
        id='data-modeling',
        title='Моделирование и нормализация',
        summary='Нормальные формы, денормализация, проектирование схем.',
        topics=['1NF/2NF/3NF', 'Star/Snowflake', 'Keys'],
        resources=[
            Resource(
                title='Intro to Data Modeling',
                url='https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/',
                type='article',
            )
        ],
        estimated_hours=4,
        tags=['modeling'],
    ),
    Milestone(
        id='practice',
        title='Практика и проекты',
        summary='Практические задачи и мини-проекты.',
        topics=['LeetSQL', 'Kaggle Datasets + SQL', 'Toy BI dashboard'],
        resources=[
            Resource(title='SQLBolt Exercises', url='https://sqlbolt.com/', type='exercise')
        ],
        estimated_hours=6,
        tags=['practice'],
    ),
]


def _infer_level_from_text(blobs: List[str]) -> str:
    text = (' '.join(blobs)[:5000]).lower()
    # Очень упрощённые эвристики
    if any(
        k in text for k in ['window function', 'окн', 'explain analyze', 'индекс', 'изоляц']
    ):
        return 'intermediate'
    if any(k in text for k in ['cte', 'подзапрос', 'recursive']):
        return 'intermediate'
    if any(k in text for k in ['partition', 'shard', 'mvcc', 'deadlock']):
        return 'advanced'
    return 'beginner'


def generate_sql_roadmap(text_blobs: List[str]) -> Roadmap:
    level = _infer_level_from_text(text_blobs)
    milestones = list(BASIC_MILESTONES)

    joined_text = ' '.join(text_blobs).lower()
    if 'postgres' in joined_text or 'postgresql' in joined_text:
        for m in milestones:
            if m.id in {'indexes-perf', 'window-funcs'}:
                m.resources.append(
                    Resource(
                        title='PG: EXPLAIN',
                        url='https://www.postgresql.org/docs/current/using-explain.html',
                        type='doc',
                    )
                )
    if 'mysql' in joined_text:
        for m in milestones:
            if m.id == 'indexes-perf':
                m.resources.append(
                    Resource(
                        title='MySQL Optimizer',
                        url='https://dev.mysql.com/doc/refman/8.0/en/optimizer-statistics.html',
                        type='doc',
                    )
                )
    if 'analytics' in joined_text or 'bi' in joined_text or 'аналит' in joined_text:
        for m in milestones:
            if m.id == 'window-funcs':
                m.estimated_hours = max(m.estimated_hours, 6)
                if 'priority' not in m.tags:
                    m.tags.append('priority')

    total = sum(m.estimated_hours for m in milestones)
    return Roadmap(
        title='SQL Roadmap',
        level=level,
        total_estimated_hours=total,
        milestones=milestones,
        notes='Сгенерировано эвристиками на основе загруженных файлов (если были).',
    )


def rewrite_roadmap_with_prompt(roadmap: Roadmap, prompt: str) -> Roadmap:
    p = prompt.lower().strip()

    if 'beginner' in p or 'нович' in p or 'начина' in p:
        roadmap.level = 'beginner'
    elif 'advanced' in p or 'продвин' in p:
        roadmap.level = 'advanced'
    elif 'intermediate' in p or 'средн' in p:
        roadmap.level = 'intermediate'

    if 'postgres' in p or 'postgresql' in p:
        for m in roadmap.milestones:
            if m.id in {'indexes-perf', 'window-funcs'}:
                if not any('postgresql' in (r.url or '') for r in m.resources):
                    m.resources.append(
                        Resource(
                            title='PostgreSQL Docs',
                            url='https://www.postgresql.org/docs/current/',
                            type='doc',
                        )
                    )
    if 'mysql' in p:
        for m in roadmap.milestones:
            if m.id == 'indexes-perf':
                if not any('mysql' in (r.url or '') for r in m.resources):
                    m.resources.append(
                        Resource(
                            title='MySQL 8.0 Reference',
                            url='https://dev.mysql.com/doc/refman/8.0/en/',
                            type='doc',
                        )
                    )

    weeks_match = re.search(r'за\s+(\d+)\s*нед', p)
    hours_match = re.search(r'(?:<=|≤|не более|до)\s*(\d+)\s*час', p)
    if weeks_match:
        weeks = int(weeks_match.group(1))
        target = weeks * 7
        _scale_hours(roadmap, target)
    elif hours_match:
        target = int(hours_match.group(1))
        _scale_hours(roadmap, target)

    if 'добавь ресурсы' in p or 'more resources' in p or 'extra resources' in p:
        for m in roadmap.milestones:
            m.resources.extend(
                [
                    Resource(title='SQLZoo', url='https://sqlzoo.net/', type='exercise'),
                    Resource(
                        title='w3schools SQL', url='https://www.w3schools.com/sql/', type='doc'
                    ),
                ]
            )

    rm = re.findall(r'(?:убери|remove|исключи)\s+([a-zа-я0-9\-\s]+)', p)
    if rm:
        bad_kw = {x.strip().lower() for x in rm}
        roadmap.milestones = [
            m
            for m in roadmap.milestones
            if all(kw not in m.title.lower() and kw not in m.summary.lower() for kw in bad_kw)
        ]

    if any(k in p for k in ['аналит', 'analytics', 'bi', 'data analyst']):
        for m in roadmap.milestones:
            if 'analytics' in m.tags or m.id == 'window-funcs':
                if 'priority' not in m.tags:
                    m.tags.append('priority')
                m.estimated_hours = max(m.estimated_hours, 6)

    roadmap.total_estimated_hours = sum(m.estimated_hours for m in roadmap.milestones)
    return roadmap


def _scale_hours(roadmap: Roadmap, target_total: int) -> None:
    current = sum(m.estimated_hours for m in roadmap.milestones)
    if current == 0:
        return
    scale = target_total / current
    updated = []
    for m in roadmap.milestones:
        new_h = max(1, round(m.estimated_hours * scale))
        m.estimated_hours = new_h
        updated.append(m)
    roadmap.milestones = updated
