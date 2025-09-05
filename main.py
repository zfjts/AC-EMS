import json
import uuid
import shutil
from fastapi import FastAPI, Depends, HTTPException, Form, Request, UploadFile, File, Response
from fastapi.responses import HTMLResponse, RedirectResponse, Response, FileResponse, JSONResponse
import pandas as pd
import os
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends, File, Form, UploadFile, HTTPException
import os
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Table, func, extract
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session, joinedload, relationship
from pydantic import BaseModel
from datetime import datetime, date, timedelta
from typing import List, Optional, Union
from fastapi import FastAPI, Request, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, Table, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from sqlalchemy.exc import SQLAlchemyError
import os
import pandas as pd
from io import BytesIO
from dateutil.relativedelta import relativedelta

import os
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Table
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session, joinedload, relationship
from pydantic import BaseModel
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import uvicorn
import os
from fastapi import Cookie
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

# 配置数据库
# 确保数据库目录存在
import os
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///" + os.path.join(os.path.dirname(__file__), 'data', 'database_new.db')
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 初始化FastAPI应用
app = FastAPI(title="自控设备全生命周期管理平台")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# 添加月份过滤器
from datetime import datetime

def add_months(date_str, months):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.month + months - 1
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31, 29 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
    return datetime(year, month, day).strftime('%Y-%m-%d')

templates.env.filters['add_months'] = add_months

@app.get("/@vite/client")
async def vite_client():
    return Response(status_code=204)

# 模板上下文依赖项
async def get_template_context(request: Request, db: Session = Depends(get_db)):
    try:
        current_user = await get_current_user(request, db)
        return {
            "request": request,
            "current_user": current_user
        }
    except:
        return {
            "request": request,
            "current_user": None
        }

# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 配置
SECRET_KEY = "your-secret-key-here"  # 实际应用中应使用更安全的密钥并存储在环境变量中
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 密码哈希和验证函数
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 创建访问令牌
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 获取当前用户
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = db.query(UserDB).filter(UserDB.username == username).first()
        return user
    except JWTError:
        return None

# 确保用户已登录
def get_current_active_user(current_user: "UserDB" = Depends(get_current_user)):
    if current_user is None:
        raise HTTPException(status_code=307, detail="需要登录", headers={"Location": "/login/"})
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户已禁用")
    return current_user

# 角色验证依赖项
def get_current_admin_user(current_user: "UserDB" = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user

# 写权限验证依赖项
def get_write_permission_user(current_user: "UserDB" = Depends(get_current_active_user)):
    if current_user.role == "readonly":
        raise HTTPException(status_code=403, detail="只读用户无操作权限")
    return current_user

# 设备模型
class EquipmentDB(Base):
    __tablename__ = "equipment"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    model = Column(String)
    production_number = Column(String, unique=True, nullable=True)
    manufacturer = Column(String, nullable=True)
    installation_date = Column(DateTime, nullable=True)
    location = Column(String)
    status = Column(String)  # 运行状态
    equipment_type = Column(String)  # 设备类型
    working_life = Column(Float)  # 工作寿命(月)
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    decommission_date = Column(DateTime, nullable=True)  # 退役日期
    parameters = Column(String, nullable=True)  # 存储参数的JSON字符串
    attachments = Column(String, nullable=True)  # 存储附属信息的JSON字符串
    documents = relationship("DocumentDB", secondary="document_equipment", back_populates="equipment")

# 用户模型
class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    email = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    role = Column(String, default="user")  # admin, user, readonly
    created_at = Column(DateTime, default=datetime.now)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

# 配件管理相关模型
class LocationDB(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    code = Column(String(50), nullable=False, unique=True)
    category = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class SparePartDB(Base):
    __tablename__ = "spare_parts"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    code = Column(String(50), nullable=False, unique=True)
    specification = Column(String(255))
    unit = Column(String(20), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"))
    safety_stock = Column(Integer, default=0)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    location = relationship("LocationDB", backref="spare_parts")

class InventoryDB(Base):
    __tablename__ = "inventory"
    id = Column(Integer, primary_key=True, index=True)
    spare_part_id = Column(Integer, ForeignKey("spare_parts.id"), nullable=False)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    quantity = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    spare_part = relationship("SparePartDB", backref="inventory_records")
    location = relationship("LocationDB", backref="inventory_records")

class InventoryTransactionDB(Base):
    __tablename__ = "inventory_transactions"
    id = Column(Integer, primary_key=True, index=True)
    spare_part_id = Column(Integer, ForeignKey("spare_parts.id"), nullable=False)
    from_location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)
    to_location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)
    quantity = Column(Integer, nullable=False)
    transaction_type = Column(String(20), nullable=False)  # 入库、出库、移库
    order_number = Column(String(50))
    status = Column(String(20), default="保存")  # 保存、提交
    operator = Column(String(50))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    spare_part = relationship("SparePartDB")
    from_location = relationship("LocationDB", foreign_keys=[from_location_id])
    to_location = relationship("LocationDB", foreign_keys=[to_location_id])

# 故障报告模型
class FaultReportDB(Base):
    __tablename__ = "fault_reports"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"))
    report_date = Column(DateTime)
    fault_type = Column(String)
    description = Column(String)
    status = Column(String, default="pending")  # pending, processing, resolved
    image_path = Column(String, nullable=True)
    created_by = Column(String)
    resolved_date = Column(DateTime, nullable=True)
    resolution = Column(String, nullable=True)
    cost = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)  # 耗时(小时)

    equipment = relationship("EquipmentDB", backref="fault_reports")

# 出入库管理路由
@app.get("/inventory/transactions/", response_class=HTMLResponse)
async def list_inventory_transactions(request: Request, db: Session = Depends(get_db)):
    transactions = db.query(InventoryTransactionDB).options(
        joinedload(InventoryTransactionDB.spare_part),
        joinedload(InventoryTransactionDB.from_location),
        joinedload(InventoryTransactionDB.to_location)
    ).order_by(InventoryTransactionDB.created_at.desc()).all()
    return templates.TemplateResponse("inventory_transaction.html", {
        "request": request, "transactions": transactions
    })

@app.get("/inventory/transactions/create/", response_class=HTMLResponse)
async def create_inventory_transaction(request: Request, db: Session = Depends(get_db)):
    transaction_type = request.query_params.get("type", "入库")
    spare_parts = db.query(SparePartDB).all()
    locations = db.query(LocationDB).all()
    return templates.TemplateResponse("inventory_transaction_form.html", {
        "request": request, "transaction_type": transaction_type,
        "spare_parts": spare_parts, "locations": locations, "transaction": None
    })

@app.post("/inventory/transactions/create/")
async def save_inventory_transaction(
    request: Request, db: Session = Depends(get_db), _: UserDB = Depends(get_write_permission_user)
):
    form = await request.form()
    transaction_type = form.get("transaction_type", "入库")
    action = form.get("action", "save")
    
    transaction = InventoryTransactionDB(
        spare_part_id=int(form.get("spare_part_id")),
        from_location_id=int(form.get("from_location_id")) if form.get("from_location_id") else None,
        to_location_id=int(form.get("to_location_id")) if form.get("to_location_id") else None,
        quantity=int(form.get("quantity")),
        transaction_type=transaction_type,
        order_number=form.get("order_number"),
        status="提交" if action == "submit" else "保存",
        operator=form.get("operator"),
        notes=form.get("notes")
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    
    # 如果提交，更新库存
    if action == "submit":
        if transaction_type == "入库":
            inventory = db.query(InventoryDB).filter(
                InventoryDB.spare_part_id == transaction.spare_part_id,
                InventoryDB.location_id == transaction.to_location_id
            ).first()
            if inventory:
                inventory.quantity += transaction.quantity
            else:
                inventory = InventoryDB(
                    spare_part_id=transaction.spare_part_id,
                    location_id=transaction.to_location_id,
                    quantity=transaction.quantity
                )
                db.add(inventory)
        elif transaction_type == "出库":
            inventory = db.query(InventoryDB).filter(
                InventoryDB.spare_part_id == transaction.spare_part_id,
                InventoryDB.location_id == transaction.from_location_id
            ).first()
            if inventory and inventory.quantity >= transaction.quantity:
                inventory.quantity -= transaction.quantity
            else:
                # 库存不足，这里可以添加错误处理
                pass
        db.commit()
    
    return RedirectResponse(url="/inventory/transactions/", status_code=303)

@app.get("/inventory/transactions/edit/{transaction_id}", response_class=HTMLResponse)
async def edit_inventory_transaction(request: Request, transaction_id: int, db: Session = Depends(get_db)):
    transaction = db.query(InventoryTransactionDB).filter(InventoryTransactionDB.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="交易记录未找到")
    spare_parts = db.query(SparePartDB).all()
    locations = db.query(LocationDB).all()
    return templates.TemplateResponse("inventory_transaction_form.html", {
        "request": request, "transaction_type": transaction.transaction_type,
        "spare_parts": spare_parts, "locations": locations, "transaction": transaction
    })

@app.post("/inventory/transactions/edit/{transaction_id}")
async def update_inventory_transaction(
    request: Request, transaction_id: int, db: Session = Depends(get_db), _: UserDB = Depends(get_write_permission_user)
):
    # 类似创建逻辑，但更新现有交易
    transaction = db.query(InventoryTransactionDB).filter(InventoryTransactionDB.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="交易记录未找到")
    
    form = await request.form()
    action = form.get("action", "save")
    
    # 这里应该先回滚之前的库存变更（如果已提交）
    # 简化处理，实际应用中需要更复杂的逻辑
    
    transaction.spare_part_id = int(form.get("spare_part_id"))
    transaction.from_location_id = int(form.get("from_location_id")) if form.get("from_location_id") else None
    transaction.to_location_id = int(form.get("to_location_id")) if form.get("to_location_id") else None
    transaction.quantity = int(form.get("quantity"))
    transaction.order_number = form.get("order_number")
    transaction.status = "提交" if action == "submit" else "保存"
    transaction.operator = form.get("operator")
    transaction.notes = form.get("notes")
    
    db.commit()
    
    # 如果提交，更新库存
    if action == "submit":
        # 简化处理，实际应用中需要更复杂的逻辑
        pass
    
    return RedirectResponse(url="/inventory/transactions/", status_code=303)

@app.get("/inventory/transactions/submit/{transaction_id}")
async def submit_inventory_transaction(request: Request, transaction_id: int, db: Session = Depends(get_db)):
    transaction = db.query(InventoryTransactionDB).filter(InventoryTransactionDB.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="交易记录未找到")
    
    transaction.status = "提交"
    db.commit()
    
    # 更新库存
    if transaction.transaction_type == "入库":
        inventory = db.query(InventoryDB).filter(
            InventoryDB.spare_part_id == transaction.spare_part_id,
            InventoryDB.location_id == transaction.to_location_id
        ).first()
        if inventory:
            inventory.quantity += transaction.quantity
        else:
            inventory = InventoryDB(
                spare_part_id=transaction.spare_part_id,
                location_id=transaction.to_location_id,
                quantity=transaction.quantity
            )
            db.add(inventory)
    elif transaction.transaction_type == "出库":
        inventory = db.query(InventoryDB).filter(
            InventoryDB.spare_part_id == transaction.spare_part_id,
            InventoryDB.location_id == transaction.from_location_id
        ).first()
        if inventory and inventory.quantity >= transaction.quantity:
            inventory.quantity -= transaction.quantity
        else:
            # 库存不足，这里可以添加错误处理
            pass
    db.commit()
    
    return RedirectResponse(url="/inventory/transactions/", status_code=303)

@app.get("/inventory/transactions/delete/{transaction_id}")
async def delete_inventory_transaction(request: Request, transaction_id: int, db: Session = Depends(get_db)):
    transaction = db.query(InventoryTransactionDB).filter(InventoryTransactionDB.id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="交易记录未找到")
    
    # 如果已提交，回滚库存
    if transaction.status == "提交":
        # 简化处理，实际应用中需要更复杂的逻辑
        pass
    
    db.delete(transaction)
    db.commit()
    
    return RedirectResponse(url="/inventory/transactions/", status_code=303)

# 库存优化分析路由
@app.get("/inventory/analysis/", response_class=HTMLResponse)
async def inventory_analysis(request: Request, db: Session = Depends(get_db)):
    # 简化的分析数据，实际应用中需要更复杂的计算
    
    # 库存周转率 (实际计算)
    transactions = db.query(InventoryTransactionDB).filter(
        InventoryTransactionDB.transaction_type.in_(['入库', '出库']),
        InventoryTransactionDB.status == '提交'
    ).all()
    total_quantity = sum(t.quantity for t in transactions)
    inventory = db.query(InventoryDB).all()
    avg_inventory = sum(i.quantity for i in inventory) / len(inventory) if inventory else 0
    turnover_rate = total_quantity / avg_inventory if avg_inventory > 0 else 0
    
    # 计算周转率趋势 (与上月比较)
    last_month = datetime.now().replace(day=1) - timedelta(days=1)
    last_month_start = datetime(last_month.year, last_month.month, 1)
    last_month_transactions = db.query(InventoryTransactionDB).filter(
        InventoryTransactionDB.transaction_type.in_(['入库', '出库']),
        InventoryTransactionDB.status == '提交',
        InventoryTransactionDB.created_at >= last_month_start,
        InventoryTransactionDB.created_at <= last_month
    ).all()
    last_month_total = sum(t.quantity for t in last_month_transactions)
    last_month_inventory = db.query(InventoryDB).filter(
        InventoryDB.updated_at >= last_month_start,
        InventoryDB.updated_at <= last_month
    ).all()
    last_month_avg = sum(i.quantity for i in last_month_inventory) / len(last_month_inventory) if last_month_inventory else 0
    last_month_turnover = last_month_total / last_month_avg if last_month_avg > 0 else 0
    turnover_trend = ((turnover_rate - last_month_turnover) / last_month_turnover * 100) if last_month_turnover > 0 else 0
    turnover_trend = f"{turnover_trend:.1f}"
    
    # 资金占用 (实际计算)
    spare_parts = db.query(SparePartDB).all()
    part_dict = {p.id: p for p in spare_parts}
    capital_occupy = sum(i.quantity * part_dict[i.spare_part_id].price for i in inventory if i.spare_part_id in part_dict)
    
    # 计算资金占用趋势 (与上月比较)
    last_month_capital = sum(i.quantity * part_dict[i.spare_part_id].price for i in last_month_inventory if i.spare_part_id in part_dict)
    capital_trend = ((capital_occupy - last_month_capital) / last_month_capital * 100) if last_month_capital > 0 else 0
    capital_trend = f"{capital_trend:.1f}"
    
    # 库存预警 (实际计算)
    low_stock_count = 0
    overstock_count = 0
    for item in inventory:
        part = part_dict.get(item.spare_part_id)
        if not part:
            continue
        # 低库存预警
        if item.quantity < part.min_stock:
            low_stock_count += 1
        # 积压库存预警 (假设6个月未使用定义为积压)
        six_months_ago = datetime.now() - timedelta(days=180)
        recent_usage = db.query(InventoryTransactionDB).filter(
            InventoryTransactionDB.spare_part_id == item.spare_part_id,
            InventoryTransactionDB.transaction_type == '出库',
            InventoryTransactionDB.status == '提交',
            InventoryTransactionDB.created_at >= six_months_ago
        ).first()
        if not recent_usage and item.quantity > part.max_stock:
            overstock_count += 1
    warning_count = low_stock_count + overstock_count
    
    # 使用频率分析数据 (实际数据)
    part_usage = {}    
    for t in transactions:
        if t.spare_part_id in part_usage:
            part_usage[t.spare_part_id] += t.quantity
        else:
            part_usage[t.spare_part_id] = t.quantity
    # 按使用频率排序并取前5名
    sorted_parts = sorted(part_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    usage_labels = [part_dict[pid].name for pid, _ in sorted_parts if pid in part_dict]
    usage_data = [qty for _, qty in sorted_parts]
    
    # 消耗趋势分析数据 (实际数据)
    # 获取最近6个月的数据
    months = []
    trend_data = []
    for i in range(6):
        month_date = datetime.now() - timedelta(days=i*30)
        month_start = datetime(month_date.year, month_date.month, 1)
        month_end = (month_start.replace(month=month_start.month+1) - timedelta(days=1)) if month_start.month < 12 else datetime(month_start.year, 12, 31)
        months.append(f"{month_start.month}月")
        
        month_transactions = db.query(InventoryTransactionDB).filter(
            InventoryTransactionDB.transaction_type == '出库',
            InventoryTransactionDB.status == '提交',
            InventoryTransactionDB.created_at >= month_start,
            InventoryTransactionDB.created_at <= month_end
        ).all()
        month_total = sum(t.quantity for t in month_transactions)
        trend_data.append(month_total)
    # 反转以按时间顺序显示
    trend_labels = months[::-1]
    trend_data = trend_data[::-1]
    
    # 库存优化建议 (基于实际数据)
    optimization_suggestions = []
    # 低库存建议
    if low_stock_count > 0:
        optimization_suggestions.append({
            "title": f"补充{low_stock_count}个低库存备件",
            "content": f"有{low_stock_count}个备件库存低于安全阈值，建议及时采购补充。",
            "level": "red",
            "parts": [part_dict[item.spare_part_id].name for item in inventory if item.spare_part_id in part_dict and item.quantity < part_dict[item.spare_part_id].min_stock][:5]
        })
    # 积压库存建议
    if overstock_count > 0:
        optimization_suggestions.append({
            "title": f"处理{overstock_count}个积压备件",
            "content": f"有{overstock_count}个备件长期未使用且库存过高，建议调整采购策略或促销。",
            "level": "yellow",
            "parts": [part_dict[item.spare_part_id].name for item in inventory if item.spare_part_id in part_dict and item.quantity > part_dict[item.spare_part_id].max_stock][:5]
        })
    
    spare_parts = db.query(SparePartDB).all()
    
    return templates.TemplateResponse("inventory_analysis.html", {
        "request": request,
        "turnover_rate": turnover_rate,
        "turnover_trend": turnover_trend,
        "capital_occupy": capital_occupy,
        "capital_trend": capital_trend,
        "warning_count": warning_count,
        "low_stock_count": low_stock_count,
        "overstock_count": overstock_count,
        "usage_labels": usage_labels,
        "usage_data": usage_data,
        "trend_labels": trend_labels,
        "trend_data": trend_data,
        "optimization_suggestions": optimization_suggestions,
        "spare_parts": spare_parts
    })

# 设备参数路由
@app.get("/equipment/{equipment_id}/parameters")
async def equipment_parameters(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).options(joinedload(EquipmentDB.documents)).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析参数（假设存储为JSON字符串）
    import json
    parameters = json.loads(equipment.parameters) if equipment.parameters else {}
    
    # 删除模拟参数记录数据
    parameter_records = []
    
    return templates.TemplateResponse("equipment_parameters.html", {
        "request": request,
        "equipment": equipment,
        "parameters": parameters,
        "parameter_records": parameter_records
    })

# 添加参数记录路由
@app.post("/equipment/{equipment_id}/parameters/add")
async def add_parameter_record(
    equipment_id: int,
    param_name: str = Form(...),
    param_value: str = Form(...),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有参数记录
    import json
    parameters = json.loads(equipment.parameters) if equipment.parameters else {}
    parameter_records = parameters.get("records", [])
    
    # 添加新参数记录
    new_record = {
        "param_name": param_name,
        "param_value": param_value,
        "modified_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    parameter_records.append(new_record)
    
    # 更新参数数据
    parameters["records"] = parameter_records
    equipment.parameters = json.dumps(parameters)
    db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/parameters", status_code=303)

# 设备附属信息路由
@app.get("/equipment/{equipment_id}/attachments")
async def equipment_attachments(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析附属信息（假设存储为JSON字符串）
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    
    # 从解析的数据中获取附属设备和文档
    attached_equipment = attachments_data.get("equipment", [])
    attached_documents = attachments_data.get("documents", [])
    
    attachments = {
        "equipment": attached_equipment,
        "documents": attached_documents
    }
    
    return templates.TemplateResponse("equipment_attachments.html", {
        "request": request,
        "equipment": equipment,
        "attachments": attachments
    })

# 添加附属设备路由
@app.post("/equipment/{equipment_id}/attachments/add_equipment")
async def add_attached_equipment(
    equipment_id: int,
    name: str = Form(...),
    model: str = Form(None),
    manufacturer: str = Form(None),
    relation_type: str = Form(...),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    print(f"设备ID: {equipment_id}, 文档ID: {doc_id}")
    print(f"设备附属信息: {equipment.attachments}")
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    print(f"解析后的附属信息: {attachments_data}")
    attached_equipment = attachments_data.get("equipment", [])
    
    # 添加新附属设备
    new_equipment = {
        "name": name,
        "model": model,
        "manufacturer": manufacturer,
        "relation_type": relation_type
    }
    attached_equipment.append(new_equipment)
    
    # 更新数据库
    attachments_data["equipment"] = attached_equipment
    equipment.attachments = json.dumps(attachments_data)
    db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 删除附属设备路由
@app.get("/equipment/{equipment_id}/attachments/delete_equipment/{index}")
async def delete_attached_equipment(equipment_id: int, index: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    attached_equipment = attachments_data.get("equipment", [])
    
    # 检查索引是否有效
    if 0 <= index < len(attached_equipment):
        attached_equipment.pop(index)
        attachments_data["equipment"] = attached_equipment
        equipment.attachments = json.dumps(attachments_data)
        db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 添加附属文档路由
@app.post("/equipment/{equipment_id}/attachments/add_document")
async def add_attached_document(
    equipment_id: int,
    doc_name: str = Form(...),
    doc_type: str = Form(...),
    doc_description: str = Form(None),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")

    # 解析现有的attachments数据
    attachments = {}
    if equipment.attachments:
        try:
            attachments = json.loads(equipment.attachments)
        except json.JSONDecodeError:
            pass

    # 确保documents数组存在
    if 'documents' not in attachments:
        attachments['documents'] = []

    # 添加新文档
    new_doc = {
        'id': len(attachments['documents']) + 1,
        'name': doc_name,
        'type': doc_type,
        'description': doc_description,
        'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    attachments['documents'].append(new_doc)

    # 更新设备的attachments字段
    equipment.attachments = json.dumps(attachments)
    db.commit()

    # 同时创建DocumentDB记录
    db_document = DocumentDB(
        title=doc_name,
        file_name=f"{doc_name}.{doc_type.lower()}",
        file_path="",  # 实际应用中应该上传文件并保存路径
        file_type=f"document/{doc_type.lower()}",
        category="设备附属文档",
        upload_date=datetime.now(),
        description=doc_description
    )
    db_document.equipment.append(equipment)
    db.add(db_document)
    db.commit()

    return RedirectResponse(url=f"/equipment/{equipment_id}/attachments", status_code=303)

@app.get("/equipment/{equipment_id}/attachments/delete_document/{doc_id}")
async def delete_attached_document(equipment_id: int, doc_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    attached_documents = attachments_data.get("documents", [])
    
    # 查找并删除指定ID的文档
    print(f"删除前的文档列表: {attached_documents}")
    
    # 在删除前获取要删除的文档信息
    doc_to_delete = next((d for d in attached_documents if d.get('id') == doc_id), None)
    
    # 删除文档
    attached_documents = [doc for doc in attached_documents if doc.get('id') != doc_id]
    print(f"删除后的文档列表: {attached_documents}")
    attachments_data["documents"] = attached_documents
    equipment.attachments = json.dumps(attachments_data)
    
    # 同时删除DocumentDB中的对应记录
    if doc_to_delete:
        db_document = db.query(DocumentDB).filter(
            DocumentDB.title == doc_to_delete['name'],
            DocumentDB.equipment.any(id=equipment_id)
        ).first()
        if db_document:
            db.delete(db_document)
            
    db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 批量上传相关路由
@app.get("/equipment/batch/upload/")
async def batch_upload_equipment_form(request: Request):
    return templates.TemplateResponse("equipment_batch_form.html", {"request": request})

@app.post("/equipment/batch/upload/")
async def batch_upload_equipment(
    request: Request,
    file: UploadFile = File(...),
    skip_errors: bool = Form(False),
    override_existing: bool = Form(False),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    # 验证文件类型
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="请上传Excel文件 (.xlsx)")

    try:
        # 读取Excel文件
        df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {str(e)}")

    # 验证必要列
    required_columns = ['name', 'model', 'location', 'status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"缺少必要列: {', '.join(missing_columns)}")

    # 初始化结果统计
    success_count = 0
    error_count = 0
    errors = []

    # 处理每一行数据
    for index, row in df.iterrows():
        row_num = index + 2  # 行号从2开始（第一行是表头）
        try:
            # 验证状态值
            if row['status'] not in ['未安装', '运行中']:
                raise ValueError("状态必须是'未安装'或'运行中'")

            # 转换日期
            installation_date = None
            if 'installation_date' in row and pd.notna(row['installation_date']) and str(row['installation_date']).strip():
                try:
                    installation_date = datetime.strptime(str(row['installation_date']).strip(), "%Y-%m-%d")
                except ValueError:
                    raise ValueError("安装日期格式不正确，请使用YYYY-MM-DD格式")

            # 检查生产编号是否已存在
            production_number = row.get('production_number')
            existing_equipment = None

            if production_number:
                existing_equipment = db.query(EquipmentDB).filter(EquipmentDB.production_number == production_number).first()

            # 如果存在且不覆盖，则跳过
            if existing_equipment and not override_existing:
                error_count += 1
                errors.append({
                    'row': row_num,
                    'data': {'name': row['name'], 'model': row['model']},
                    'error': f"生产编号'{production_number}'已存在，且未选择覆盖"
                })
                continue

            # 计算下次维护时间（假设安装后30天首次维护）
            next_maintenance = None
            if installation_date:
                next_maintenance = installation_date.replace(day=min(installation_date.day + 30, 28))

            if existing_equipment:
                # 更新现有设备
                existing_equipment.name = row['name']
                existing_equipment.model = row['model']
                existing_equipment.manufacturer = row.get('manufacturer')
                existing_equipment.installation_date = installation_date
                existing_equipment.location = row['location']
                existing_equipment.status = row['status']
                existing_equipment.equipment_type = row.get('equipment_type')
                existing_equipment.next_maintenance = next_maintenance
                db.commit()
            else:
                # 创建新设备
                db_equipment = EquipmentDB(
                    name=row['name'],
                    model=row['model'],
                    production_number=production_number if pd.notna(production_number) else None,
                    manufacturer=row['manufacturer'],
                    installation_date=installation_date,
                    location=row['location'],
                    status=row['status'],
                    equipment_type=row.get('equipment_type'),
                    next_maintenance=next_maintenance
                )
                db.add(db_equipment)
                db.commit()

            success_count += 1
        except Exception as e:
            error_count += 1
            errors.append({
                'row': row_num,
                'data': {'name': row.get('name', '未知'), 'model': row.get('model', '未知')},
                'error': str(e)
            })
            if not skip_errors:
                raise HTTPException(status_code=400, detail=f"第{row_num}行数据错误: {str(e)}")

    # 提交所有更改
    db.commit()

    # 返回结果页面
    return templates.TemplateResponse("equipment_batch_result.html", {
        "request": request,
        "success_count": success_count,
        "error_count": error_count,
        "errors": errors
    })

# 模板下载路由
@app.get("/equipment/template/download/")
async def download_equipment_template():
    # 创建模板数据
    template_data = [
        {'name': '设备名称', 'model': '型号', 'production_number': '生产编号(可选)', 'manufacturer': '制造商', 'installation_date': '安装日期(YYYY-MM-DD)', 'location': '位置', 'status': '状态(未安装/运行中)'},
        {'name': '示例设备1', 'model': 'Model-X', 'production_number': 'SN123456', 'manufacturer': '示例厂商', 'installation_date': '2023-01-15', 'location': '车间A', 'status': '运行中'},
        {'name': '示例设备2', 'model': 'Model-Y', 'production_number': '', 'manufacturer': '示例厂商', 'installation_date': '2023-02-20', 'location': '车间B', 'status': '未安装'}
    ]

    # 创建Excel文件
    df = pd.DataFrame(template_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='设备模板')

    output.seek(0)

    # 设置缓存控制头，防止浏览器缓存旧版本
    headers = {
        'Content-Disposition': 'attachment; filename="equipment_template.xlsx"',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    return Response(content=output.getvalue(), headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    

# 数据导出路由
@app.get("/equipment/export/")
async def export_equipment_data(db: Session = Depends(get_db)):
    # 查询所有设备数据
    equipments = db.query(EquipmentDB).all()

    # 创建数据列表
    data = []
    for eq in equipments:
        data.append({
            'id': eq.id,
            'name': eq.name,
            'model': eq.model,
            'production_number': eq.production_number if eq.production_number else '',
            'manufacturer': eq.manufacturer,
            'installation_date': eq.installation_date.strftime('%Y-%m-%d'),
            'location': eq.location,
            'status': eq.status,
            'next_maintenance': eq.next_maintenance.strftime('%Y-%m-%d') if eq.next_maintenance else ''
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 创建临时文件
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    temp_file = os.path.join(temp_dir, f"equipment_data_{timestamp}.xlsx")

    # 写入Excel文件
    df.to_excel(temp_file, index=False)

    # 设置缓存控制头，防止浏览器缓存旧版本
    headers = {
        'Content-Disposition': f'attachment; filename="equipment_data_{datetime.now().strftime('%Y%m%d')}.xlsx"',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    return FileResponse(temp_file, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 添加自定义过滤器到Jinja2模板环境
def add_months_filter(date_str, months):
    # 将字符串日期转换为datetime对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # 添加指定的月数
    new_date = date_obj + relativedelta(months=months)
    # 格式化为字符串返回
    return new_date.strftime("%Y-%m-%d")

# 注册过滤器
templates.env.filters['add_months'] = add_months_filter

# 维护统计分析路由
@app.get("/maintenance/statistics/", response_class=HTMLResponse)
async def maintenance_statistics(request: Request, db: Session = Depends(get_db)):
    # 获取当前日期
    today = date.today()
    current_month = today.month
    current_year = today.year
    
    # 计算上个月的起始和结束日期
    last_month = current_month - 1 if current_month > 1 else 12
    last_year = current_year if current_month > 1 else current_year - 1
    start_of_last_month = date(last_year, last_month, 1)
    end_of_last_month = date(current_year, current_month, 1) - timedelta(days=1)
    
    # 计算当前月的起始日期
    start_of_current_month = date(current_year, current_month, 1)
    
    # 获取所有维护记录
    all_maintenances = db.query(MaintenanceDB).all()
    total_maintenances = len(all_maintenances)
    
    # 获取本月维护记录
    current_month_maintenances = db.query(MaintenanceDB).filter(
        func.extract('year', MaintenanceDB.maintenance_date) == current_year,
        func.extract('month', MaintenanceDB.maintenance_date) == current_month
    ).all()
    
    # 获取上月维护记录
    last_month_maintenances = db.query(MaintenanceDB).filter(
        MaintenanceDB.maintenance_date >= start_of_last_month,
        MaintenanceDB.maintenance_date <= end_of_last_month
    ).all()
    
    # 计算总维护增长率
    last_month_count = len(last_month_maintenances)
    if last_month_count > 0:
        growth_rate_total = round(((len(current_month_maintenances) - last_month_count) / last_month_count) * 100, 2)
    else:
        growth_rate_total = 100 if len(current_month_maintenances) > 0 else 0
    
    # 按类型统计维护次数
    maintenance_by_type = [
        {'maintenance_type': '预防性维护', 'count': len([m for m in all_maintenances if m.maintenance_type == '预防性维护'])},
        {'maintenance_type': '故障维修', 'count': len([m for m in all_maintenances if m.maintenance_type == '故障维修'])},
        {'maintenance_type': '日常检查', 'count': len([m for m in all_maintenances if m.maintenance_type == '日常检查'])}
    ]
    
    # 计算预防性维护增长率
    last_month_preventive = len([m for m in last_month_maintenances if m.maintenance_type == '预防性维护'])
    current_month_preventive = len([m for m in current_month_maintenances if m.maintenance_type == '预防性维护'])
    if last_month_preventive > 0:
        growth_rate_preventive = round(((current_month_preventive - last_month_preventive) / last_month_preventive) * 100, 2)
    else:
        growth_rate_preventive = 100 if current_month_preventive > 0 else 0
    
    # 计算故障维修下降率
    last_month_fault = len([m for m in last_month_maintenances if m.maintenance_type == '故障维修'])
    current_month_fault = len([m for m in current_month_maintenances if m.maintenance_type == '故障维修'])
    if last_month_fault > 0:
        decline_rate_fault = round(((last_month_fault - current_month_fault) / last_month_fault) * 100, 2)
    else:
        decline_rate_fault = 0
    
    # 计算平均维护时长
    # 注意: 假设MaintenanceDB模型添加了duration字段来存储维护时长(小时)
    all_durations = [m.duration for m in all_maintenances if hasattr(m, 'duration') and m.duration]
    avg_maintenance_time = round(sum(all_durations) / len(all_durations), 1) if all_durations else 0

    # 计算上月平均维护时长
    last_month_durations = [m.duration for m in last_month_maintenances if hasattr(m, 'duration') and m.duration]
    last_month_avg = round(sum(last_month_durations) / len(last_month_durations), 1) if last_month_durations else 0
    time_reduction = round(last_month_avg - avg_maintenance_time, 1) if last_month_avg and avg_maintenance_time else 0
    
    # 按设备统计维护次数
    equipment_maintenance_counts = db.query(
        EquipmentDB.name,
        func.count(MaintenanceDB.id).label('count')
    ).join(
        MaintenanceDB, EquipmentDB.id == MaintenanceDB.equipment_id
    ).group_by(
        EquipmentDB.name
    ).all()
    maintenance_by_equipment = [{'name': eq.name, 'count': eq.count} for eq in equipment_maintenance_counts]
    
    # 准备趋势分析数据 (近6个月)
    trend_labels = []
    preventive_maintenance_data = []
    fault_maintenance_data = []
    
    for i in range(6):
        month = current_month - i if current_month - i > 0 else 12 + (current_month - i)
        year = current_year if current_month - i > 0 else current_year - 1
        
        # 格式化月份标签
        trend_labels.append(f'{year}-{month:02d}')
        
        # 计算该月的起始和结束日期
        month_start = date(year, month, 1)
        next_month = month + 1 if month < 12 else 1
        next_month_year = year if month < 12 else year + 1
        month_end = date(next_month_year, next_month, 1) - timedelta(days=1)
        
        # 统计该月的预防性维护和故障维修次数
        preventive_count = db.query(MaintenanceDB).filter(
            MaintenanceDB.maintenance_date >= month_start,
            MaintenanceDB.maintenance_date <= month_end,
            MaintenanceDB.maintenance_type == '预防性维护'
        ).count()
        
        fault_count = db.query(MaintenanceDB).filter(
            MaintenanceDB.maintenance_date >= month_start,
            MaintenanceDB.maintenance_date <= month_end,
            MaintenanceDB.maintenance_type == '故障维修'
        ).count()
        
        preventive_maintenance_data.append(preventive_count)
        fault_maintenance_data.append(fault_count)
    
    # 反转顺序，使最近的月份在右侧
    trend_labels.reverse()
    preventive_maintenance_data.reverse()
    fault_maintenance_data.reverse()
    
    # 获取维护详情
    maintenance_details = []
    recent_maintenances = db.query(MaintenanceDB).order_by(MaintenanceDB.maintenance_date.desc()).limit(10).all()
    
    for m in recent_maintenances:
        equipment = db.query(EquipmentDB).filter(EquipmentDB.id == m.equipment_id).first()
        if equipment:
            maintenance_details.append({
                'equipment': equipment.name,
                'type': m.maintenance_type,
                'date': m.maintenance_date.strftime('%Y-%m-%d'),
                'person': m.performed_by
            })
    
    # 确保trend_labels不为空
    if not trend_labels:
        trend_labels = []
        preventive_maintenance_data = []
        fault_maintenance_data = []
    
    return templates.TemplateResponse(
        "maintenance_statistics.html",
        {
            "request": request,
            "total_maintenances": total_maintenances,
            "growth_rate_total": growth_rate_total,
            "maintenance_by_type": maintenance_by_type,
            "growth_rate_preventive": growth_rate_preventive,
            "decline_rate_fault": decline_rate_fault,
            "avg_maintenance_time": avg_maintenance_time,
            "time_reduction": time_reduction,
            "maintenance_by_equipment": maintenance_by_equipment,
            "trend_labels": trend_labels,
            "preventive_maintenance_data": preventive_maintenance_data,
            "fault_maintenance_data": fault_maintenance_data,
            "maintenance_details": maintenance_details
        }
    )

# 维护记录模型
class MaintenanceDB(Base):
    __tablename__ = "maintenance"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"))
    maintenance_date = Column(DateTime)
    maintenance_type = Column(String)
    performed_by = Column(String)
    notes = Column(String)
    duration = Column(Float, default=0.0)  # 维护时长(小时)


# 文档和设备的多对多关联表
document_equipment_association = Table(
    "document_equipment",
    Base.metadata,
    Column("document_id", Integer, ForeignKey("documents.id")),
    Column("equipment_id", Integer, ForeignKey("equipment.id"))
)

# 文档模型
class DocumentDB(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    file_name = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    category = Column(String)
    upload_date = Column(DateTime)
    description = Column(String)
    equipment = relationship("EquipmentDB", secondary=document_equipment_association, back_populates="documents")


# 创建数据库表
Base.metadata.create_all(bind=engine)

# 检查并添加duration字段
from sqlalchemy import inspect

inspector = inspect(engine)
columns = inspector.get_columns('maintenance')
column_names = [col['name'] for col in columns]

if 'duration' not in column_names:
    with engine.begin() as conn:
        conn.execute('ALTER TABLE maintenance ADD COLUMN duration REAL DEFAULT 0.0;')


# Pydantic模型用于请求和响应
# 故障报告Pydantic模型
class FaultReportBase(BaseModel):
    equipment_id: int
    report_date: datetime
    fault_type: str
    description: str
    status: str = "pending"
    created_by: str = "****"

class FaultReportCreate(FaultReportBase):
    pass

class FaultReportUpdate(BaseModel):
    status: str
    resolved_date: datetime | None = None
    resolution: str | None = None
    cost: float | None = None
    duration: float | None = None

class FaultReport(FaultReportBase):
    id: int
    image_path: str | None
    resolved_date: datetime | None
    resolution: str | None
    cost: float | None
    duration: float | None

    class Config:
        from_attributes = True

class EquipmentBase(BaseModel):
    name: str
    model: str
    production_number: str | None = None
    manufacturer: str
    installation_date: datetime
    location: str
    status: str
    equipment_type: str

class EquipmentCreate(EquipmentBase):
    pass

class Equipment(EquipmentBase):
    id: int
    last_maintenance: datetime | None
    next_maintenance: datetime | None
    decommission_date: datetime | None
    current_status: str
    equipment_type: str

    class Config:
        from_attributes = True

class MaintenanceReminder(BaseModel):
    equipment_id: int
    equipment_name: str
    model: str
    next_maintenance: datetime
    remaining_days: int
    current_status: str

    class Config:
        from_attributes = True

class DecommissionResponse(BaseModel):
    equipment_id: int
    equipment_name: str
    decommission_date: datetime
    status: str

    class Config:
        from_attributes = True

class MaintenanceBase(BaseModel):
    maintenance_date: datetime
    maintenance_type: str
    performed_by: str
    notes: str


class MaintenanceCreate(MaintenanceBase):
    equipment_id: int

class Maintenance(MaintenanceBase):
    id: int
    equipment_id: int

    class Config:
        from_attributes = True

# 文档Pydantic模型
class DocumentBase(BaseModel):
    title: str
    file_name: str
    file_type: str
    category: str
    description: str | None = None

class DocumentCreate(DocumentBase):
    equipment_ids: list[int] | None = None

class Document(DocumentBase):
    id: int
    file_path: str
    upload_date: datetime

    class Config:
        from_attributes = True

@app.get("/", response_class=HTMLResponse)
async def read_root(context = Depends(get_template_context), db: Session = Depends(get_db)):
    # 检查用户是否登录，如果未登录重定向到登录页面
    if context["current_user"] is None:
        return RedirectResponse(url="/login/", status_code=307)
    
    # 获取设备总数
    total_equipment = db.query(EquipmentDB).count()
    
    # 获取本月维护记录
    today = date.today()
    start_of_month = date(today.year, today.month, 1)
    monthly_maintenance = db.query(MaintenanceDB).filter(
        MaintenanceDB.maintenance_date >= start_of_month
    ).count()
    
    # 获取待维护设备 (假设30天内需要维护的设备)
    future_maintenance_date = today + timedelta(days=30)
    pending_equipment_ids = db.query(EquipmentDB.id).filter(
        EquipmentDB.next_maintenance <= future_maintenance_date,
        EquipmentDB.decommission_date == None
    ).distinct()
    pending_maintenance = pending_equipment_ids.count()
    
    # 获取文档总数
    total_documents = db.query(DocumentDB).count()
    
    # 获取故障报告总数
    total_fault_reports = db.query(FaultReportDB).count()
    
    # 获取设备状态分布
    status_distribution = {
        "运行中": db.query(EquipmentDB).filter(EquipmentDB.status == "运行中").count(),
        "即将过期": db.query(EquipmentDB).filter(EquipmentDB.status == "即将过期").count(),
        "已过期": db.query(EquipmentDB).filter(EquipmentDB.status == "已过期").count(),
        "停用": db.query(EquipmentDB).filter(EquipmentDB.status == "停用").count(),
        "未安装": db.query(EquipmentDB).filter(EquipmentDB.status == "未安装").count(),
    }
    
    # 获取近6个月维护记录
    maintenance_history = []
    for i in range(6):
        month_date = date(today.year, today.month - i, 1) if today.month - i > 0 else date(today.year - 1, 12 + (today.month - i), 1)
        next_month = date(month_date.year, month_date.month + 1, 1) if month_date.month < 12 else date(month_date.year + 1, 1, 1)
        count = db.query(MaintenanceDB).filter(
            MaintenanceDB.maintenance_date >= month_date,
            MaintenanceDB.maintenance_date < next_month
        ).count()
        month_name = month_date.strftime("%Y-%m")
        maintenance_history.append({"month": month_name, "count": count})
    
    # 按照月份升序排列
    maintenance_history.sort(key=lambda x: x["month"])
    
    # 将数据转换为JSON字符串，确保前端可以正确解析
    import json
    
    # 合并上下文数据
    context.update({
        "total_equipment": json.dumps(total_equipment),
        "monthly_maintenance": json.dumps(monthly_maintenance),
        "pending_maintenance": json.dumps(pending_maintenance),
        "total_documents": json.dumps(total_documents),
        "total_fault_reports": json.dumps(total_fault_reports),
        "status_distribution": json.dumps(status_distribution),
        "maintenance_history": json.dumps(maintenance_history)
    })
    
    return templates.TemplateResponse("index.html", context)

@app.get("/equipment/", response_class=HTMLResponse)
async def read_equipment(context = Depends(get_template_context), db: Session = Depends(get_db), search: str = None, page: int = 1):
    print(f"搜索参数: {search}")
    
    # 基础查询
    query = db.query(EquipmentDB)
    
    # 搜索过滤
    if search:
        query = query.filter(
            EquipmentDB.name.contains(search) | 
            EquipmentDB.model.contains(search) | 
            EquipmentDB.manufacturer.contains(search)
        )
    
    # 计算总数
    total_count = query.count()
    print(f"总设备数量: {total_count}")
    
    # 分页设置
    items_per_page = 30
    total_pages = (total_count + items_per_page - 1) // items_per_page  # 向上取整
    current_page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # 获取当前页数据
    equipment = query.offset((current_page - 1) * items_per_page).limit(items_per_page).all()
    
    context.update({
        "equipment": equipment, 
        "search": search,
        "current_page": current_page,
        "total_pages": total_pages,
        "items_per_page": items_per_page,
        "total_items": total_count
    })
    
    return templates.TemplateResponse("equipment_list.html", context)

@app.get("/equipment/{equipment_id}", response_class=HTMLResponse)
async def read_equipment_detail(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析附属信息中的文档
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    documents = attachments_data.get("documents", [])
    
    maintenance = db.query(MaintenanceDB).filter(MaintenanceDB.equipment_id == equipment_id).all()
    return templates.TemplateResponse("equipment_detail.html", {
        "request": request, 
        "equipment": equipment, 
        "maintenance": maintenance, 
        "documents": documents
    })

@app.get("/equipment/create/", response_class=HTMLResponse)
async def create_equipment_form(request: Request):
    return templates.TemplateResponse("equipment_form.html", {"request": request})

@app.post("/equipment/create/")
async def create_equipment(
    name: str = Form(...),
    model: str = Form(...),
    production_number: str = Form(None),
    manufacturer: str = Form(...),
    installation_date: str = Form(...),
    location: str = Form(...),
    status: str = Form(...),
    equipment_type: str = Form(...),
    working_life: int = Form(None),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    # 转换日期字符串为datetime对象
    try:
        installation_date = datetime.strptime(installation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 根据设备类型自动设置工作寿命
    if working_life is None:
        working_life = WORKING_LIFE_MAP.get(equipment_type, 60)

    # 计算下次维护时间（假设安装后30天首次维护）
    next_maintenance = None
    if installation_date:
        next_maintenance = installation_date.replace(day=min(installation_date.day + 30, 28))

    db_equipment = EquipmentDB(
            name=name,
            model=model,
            production_number=production_number,
            manufacturer=manufacturer,
            installation_date=installation_date,
            location=location,
            status=status,
            equipment_type=equipment_type,
            working_life=working_life,
            next_maintenance=next_maintenance
        )
    db.add(db_equipment)
    db.commit()
    db.refresh(db_equipment)
    return RedirectResponse(url=f"/equipment/{db_equipment.id}", status_code=303)

# 设备类型与工作寿命映射表 (月)
WORKING_LIFE_MAP = {
    'DCS': 240,
    'PLC': 180,
    '变送器': 120,
    '热电阻': 120,
    '热电偶': 60,
    '调节阀': 120,
    '切断阀': 180,
    '物位计': 180,
    '压力表': 60,
    '双金属温度计': 120,
    '分析仪表': 24,
    '流量计': 120,
    '其它': 12
}

# 设备管理相关路由
@app.get("/equipment/{equipment_id}/edit", response_class=HTMLResponse)
async def edit_equipment_form(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    print(f"访问编辑表单路由: /equipment/{equipment_id}/edit")
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    return templates.TemplateResponse("equipment_form.html", {"request": request, "equipment": equipment})

@app.post("/equipment/{equipment_id}/edit")
async def update_equipment_form(
    equipment_id: int,
    name: str = Form(...),
    model: str = Form(...),
    production_number: str = Form(None),
    manufacturer: str = Form(...),
    installation_date: str = Form(...),
    location: str = Form(...),
    status: str = Form(...),
    equipment_type: str = Form(...),
    working_life: int = Form(None),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    print(f"处理更新设备请求: /equipment/{equipment_id}/edit")
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    # 转换日期字符串为datetime对象
    try:
        installation_date = datetime.strptime(installation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 更新设备信息
    db_equipment.name = name
    db_equipment.model = model
    # 检查生产编号唯一性
    # 处理空值或"None"字符串的情况
    print(f"接收到的生产编号: {production_number}, 类型: {type(production_number)}")
    if production_number in (None, '', 'None'):
        db_equipment.production_number = None
        print("生产编号为空，设置为None")
    else:
        existing_equipment = db.query(EquipmentDB).filter(
            EquipmentDB.production_number == production_number,
            EquipmentDB.id != equipment_id
        ).first()
        if existing_equipment:
            raise HTTPException(status_code=400, detail=f"生产编号'{production_number}'已存在，请使用其他编号")
        db_equipment.production_number = production_number
    db_equipment.manufacturer = manufacturer
    db_equipment.installation_date = installation_date
    db_equipment.location = location
    db_equipment.status = status
    db_equipment.equipment_type = equipment_type
    # 根据设备类型自动设置工作寿命
    if working_life is None:
        working_life = WORKING_LIFE_MAP.get(equipment_type, 60)
    db_equipment.working_life = working_life

    db.commit()
    db.refresh(db_equipment)
    return RedirectResponse(url=f"/equipment/{equipment_id}", status_code=303)

@app.post("/equipment/{equipment_id}/delete")
async def delete_equipment(equipment_id: int, db: Session = Depends(get_db), _: UserDB = Depends(get_write_permission_user)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    db.delete(equipment)
    db.commit()
    return RedirectResponse("/equipment/", status_code=303)

@app.put("/equipment/{equipment_id}/")
async def update_equipment(equipment_id: int, equipment: EquipmentCreate, db: Session = Depends(get_db), _: UserDB = Depends(get_write_permission_user)):
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    # 更新设备信息
    for key, value in equipment.dict(exclude_unset=True).items():
        setattr(db_equipment, key, value)
    
    # 根据设备类型自动设置工作寿命（如果未提供）
    if db_equipment.working_life is None:
        db_equipment.working_life = WORKING_LIFE_MAP.get(db_equipment.equipment_type, 60)
    db.commit()
    db.refresh(db_equipment)
    return db_equipment

@app.put("/equipment/{equipment_id}/decommission/")
async def decommission_equipment(equipment_id: int, decommission_date: datetime = None, db: Session = Depends(get_db)) -> DecommissionResponse:
    """将设备标记为退役"""
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    if db_equipment.decommission_date:
        raise HTTPException(status_code=400, detail="设备已停用")

    # 如果未提供退役日期，使用当前日期
    if not decommission_date:
        decommission_date = datetime.now()

    db_equipment.decommission_date = decommission_date
    db_equipment.status = "停用"
    db.commit()
    db.refresh(db_equipment)

    return {
        "equipment_id": db_equipment.id,
        "equipment_name": db_equipment.name,
        "decommission_date": db_equipment.decommission_date,
        "status": db_equipment.status
    }

@app.get("/maintenance/create/{equipment_id}", response_class=HTMLResponse)
async def create_maintenance_form(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    return templates.TemplateResponse("maintenance_form.html", {"request": request, "equipment": equipment})

@app.get("/maintenance/create/")
async def maintenance_create_redirect(request: Request):
    return RedirectResponse(url="/equipment/", status_code=303)

@app.get("/spare-parts/maintenance/create/")
async def spare_parts_maintenance_create(request: Request, db: Session = Depends(get_db)):
    spare_parts = db.query(SparePartDB).all()
    return templates.TemplateResponse("maintenance_form.html", {"request": request, "spare_parts": spare_parts, "from_spare_parts": True, "equipment": None})

@app.post("/maintenance/create/")
async def create_maintenance(
    request: Request,
    equipment_id: int = Form(...),
    maintenance_date: str = Form(...),
    maintenance_type: str = Form(...),
    performed_by: str = Form(...),
    notes: str = Form(...),
    db: Session = Depends(get_db),
    _: UserDB = Depends(get_write_permission_user)
):
    # 转换日期字符串为datetime对象
    try:
        maintenance_date = datetime.strptime(maintenance_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 更新设备的最后维护时间和下次维护时间
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    equipment.last_maintenance = maintenance_date
    # 假设每3个月维护一次
    if equipment.next_maintenance is None or maintenance_date > equipment.next_maintenance:
        if maintenance_date:
            # 根据设备类型获取维护周期
            cycle_months = MAINTENANCE_CYCLE_MAP.get(equipment.equipment_type, 3)
            equipment.next_maintenance = maintenance_date.replace(month=min(maintenance_date.month + cycle_months, 12))
            if equipment.next_maintenance.month < maintenance_date.month:
                equipment.next_maintenance = equipment.next_maintenance.replace(year=equipment.next_maintenance.year + 1)

    # 处理备件消耗
    form = await request.form()
    spare_part_items = []
    i = 1
    while f"spare_part_id_{i}" in form:
        spare_part_id = int(form[f"spare_part_id_{i}"])
        quantity = int(form[f"quantity_{i}"])

        # 检查库存
        inventory = db.query(InventoryDB).filter(
            InventoryDB.spare_part_id == spare_part_id
        ).first()

        if not inventory or inventory.quantity < quantity:
            spare_part_name = db.query(SparePartDB).get(spare_part_id).name
            raise HTTPException(status_code=400, detail=f"备件 '{spare_part_name}' 库存不足")

        # 减少库存
        inventory.quantity -= quantity

        # 创建库存交易记录
        transaction = InventoryTransactionDB(
            spare_part_id=spare_part_id,
            from_location_id=inventory.location_id,
            quantity=quantity,
            transaction_type="出库",
            order_number=f"MNT-{maintenance_date.strftime('%Y%m%d')}-NEW",
            status="提交",
            operator=performed_by,
            notes=f"设备维护消耗: {equipment.name} ({maintenance_type})"
        )
        db.add(transaction)

        # 添加到备件项目列表
        spare_part_items.append({
            "spare_part_id": spare_part_id,
            "quantity": quantity
        })

        i += 1

    db_maintenance = MaintenanceDB(
        equipment_id=equipment_id,
        maintenance_date=maintenance_date,
        maintenance_type=maintenance_type,
        performed_by=performed_by,
        notes=notes
    )
    db.add(db_maintenance)
    db.commit()
    db.refresh(db_maintenance)
    return RedirectResponse(url=f"/equipment/{equipment_id}", status_code=303)


# 维护提醒相关路由
@app.get("/maintenance/reminders/")
async def get_maintenance_reminders(context = Depends(get_template_context), db: Session = Depends(get_db), page: int = 1):
    """获取所有设备的下一次维护时间并渲染到模板（支持分页）"""
    today = datetime.now().date()
    one_month_later = today + relativedelta(months=1)
    three_months_later = today + relativedelta(months=3)
    six_months_later = today + relativedelta(months=6)

    # 查询所有未退役设备
    query = db.query(EquipmentDB)
    query = query.filter(EquipmentDB.decommission_date == None)
    equipments = query.all()

    all_reminders = []
    for equipment in equipments:
        # 检查并更新过去的维护日期
        if equipment.next_maintenance and equipment.next_maintenance.date() < today:
            # 获取维护周期
            cycle_months = MAINTENANCE_CYCLE_MAP.get(equipment.equipment_type, 3)
            # 计算需要增加的周期数
            current_date = equipment.next_maintenance.date()
            while current_date < today:
                current_date = (datetime.combine(current_date, datetime.min.time()) + relativedelta(months=cycle_months)).date()
            # 更新下次维护时间
            equipment.next_maintenance = datetime.combine(current_date, datetime.min.time())
            db.commit()
        # 如果没有设置维护日期，尝试基于安装日期和维护周期计算
        elif equipment.installation_date and not equipment.next_maintenance:
            cycle_months = MAINTENANCE_CYCLE_MAP.get(equipment.equipment_type, 3)
            # 从安装日期开始计算
            current_date = equipment.installation_date.date()
            # 计算第一个维护日期
            first_maintenance_date = (datetime.combine(current_date, datetime.min.time()) + timedelta(days=30)).date()
            
            # 循环增加维护周期直到得到未来日期
            while first_maintenance_date < today:
                first_maintenance_date = (datetime.combine(first_maintenance_date, datetime.min.time()) + relativedelta(months=cycle_months)).date()
            
            # 设置下次维护时间
            equipment.next_maintenance = datetime.combine(first_maintenance_date, datetime.min.time())
            db.commit()

        # 只有当next_maintenance不为None时才添加到提醒列表
        if equipment.next_maintenance:
            remaining_days = (equipment.next_maintenance.date() - today).days
            all_reminders.append({
                    "equipment_id": equipment.id,
                    "equipment_name": equipment.name,
                    "model": equipment.model,
                    "next_maintenance": equipment.next_maintenance,
                    "remaining_days": remaining_days,
                    "current_status": equipment.status
                })

    # 按剩余天数排序
    all_reminders.sort(key=lambda x: x["remaining_days"])
    
    # 过滤未来3个月内需要维护的设备
    three_month_reminders = [r for r in all_reminders if r['next_maintenance'].date() <= three_months_later]
    
    # 计算分页相关数据
    total_three_month_reminders = len(three_month_reminders)
    items_per_page = 10
    total_pages = (total_three_month_reminders + items_per_page - 1) // items_per_page  # 向上取整
    
    # 确保页码在有效范围内
    current_page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    # 计算当前页数据的起始和结束索引
    start_index = (current_page - 1) * items_per_page
    end_index = start_index + items_per_page
    
    # 获取当前页的数据
    reminders_table = three_month_reminders[start_index:end_index]
    
    # 过滤未来1个月内的维护提醒（用于总提醒数）
    one_month_reminders = [r for r in all_reminders if r['next_maintenance'].date() <= one_month_later]
    
    # 过滤未来7天内的紧急提醒
    urgent_reminders = [r for r in one_month_reminders if r['remaining_days'] <= 7]

    # 计算统计数据
    total_reminders = len(one_month_reminders)
    urgent_count = len(urgent_reminders)
    percent_in_7_days = round((urgent_count / total_reminders) * 100) if total_reminders > 0 else 0

    # 查询已完成的维护记录数
    completed_maintenances = db.query(MaintenanceDB).count()
    # 计算完成率
    total_equipment = db.query(EquipmentDB).count()
    completion_rate = round((completed_maintenances / total_equipment) * 100) if total_equipment > 0 else 0
    
    # 生成未来6个月的趋势数据
    future_trend = []
    current_month = today.replace(day=1)
    for i in range(6):
        target_month = current_month + relativedelta(months=i)
        month_start = target_month.replace(day=1)
        month_end = (month_start + relativedelta(months=1)) - timedelta(days=1)
        
        # 计算该月份需要维护的设备数量
        month_count = len([r for r in all_reminders 
                          if r['next_maintenance'].date() >= month_start 
                          and r['next_maintenance'].date() <= month_end])
        
        future_trend.append({
            "month": target_month.strftime('%Y-%m'),
            "count": month_count
        })

    # 计算3个月内需要维护的设备类型分布
    equipment_type_distribution = {}
    # 创建设备ID到设备类型的映射
    equipment_type_map = {eq.id: eq.equipment_type or "未分类" for eq in equipments}
    # 统计3个月内需要维护的设备类型
    for reminder in three_month_reminders:
        eq_type = equipment_type_map.get(reminder['equipment_id'], "未分类")
        equipment_type_distribution[eq_type] = equipment_type_distribution.get(eq_type, 0) + 1

    # 渲染模板
    context.update({
        "reminders": reminders_table,  # 只显示当前页的10条记录
        "total_reminders": total_reminders,  # 未来1个月内的维护数
        "urgent_reminders": urgent_count,  # 未来7天内的紧急维护数
        "percent_in_7_days": percent_in_7_days,
        "completed_maintenances": completed_maintenances,
        "completion_rate": completion_rate,
        "future_trend": future_trend,  # 未来6个月的趋势数据
        "equipment_type_distribution": equipment_type_distribution,  # 设备类型分布数据
        # 分页相关数据
        "current_page": current_page,
        "total_pages": total_pages,
        "items_per_page": items_per_page,
        "total_items": total_three_month_reminders
    })
    
    return templates.TemplateResponse("maintenance_reminders.html", context)

# 文档管理API路由
@app.get("/documents/", response_class=HTMLResponse)
async def read_documents(context = Depends(get_template_context), category: str = None, equipment_id: int = None, db: Session = Depends(get_db)):
    query = db.query(DocumentDB).options(joinedload(DocumentDB.equipment))
    if category:
        query = query.filter(DocumentDB.category == category)
    if equipment_id:
        query = query.join(DocumentDB.equipment).filter(EquipmentDB.id == equipment_id)
    documents = query.all()

    # 获取所有设备用于筛选
    equipment_list = db.query(EquipmentDB).all()
    context.update({"documents": documents, "category": category, "equipment_id": equipment_id, "equipment_list": equipment_list})
    return templates.TemplateResponse("document_list.html", context)

@app.get("/documents/create/", response_class=HTMLResponse)
async def create_document_form(request: Request, equipment_id: int = None, db: Session = Depends(get_db)):
    equipment = None
    if equipment_id:
        equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    equipment_list = db.query(EquipmentDB).all()
    return templates.TemplateResponse("document_form.html", {"request": request, "equipment": equipment, "equipment_list": equipment_list})

@app.post("/documents/create/")
async def create_document(
    request: Request,
    title: str = Form(...),
    category: str = Form(...),
    description: str | None = Form(None),
    equipment_ids: list[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 处理equipment_ids列表
    valid_equipment_ids = []
    if equipment_ids:
        for eq_id in equipment_ids:
            try:
                if eq_id and str(eq_id).strip():
                    valid_equipment_ids.append(int(eq_id))
            except ValueError:
                continue
    # 获取关联的设备对象
    equipment_list = db.query(EquipmentDB).filter(EquipmentDB.id.in_(valid_equipment_ids)).all()
    try:
        # 创建文档目录（如果不存在）
        documents_dir = os.path.join(os.path.dirname(__file__), "documents")
        os.makedirs(documents_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(documents_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"文件已保存: {file_path}")
    except Exception as e:
        print(f"文件保存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

    # 创建文档记录
    db_document = DocumentDB(
        title=title,
        file_name=file.filename,
        file_path=file_path,
        file_type=file.content_type,
        category=category,
        upload_date=datetime.now(),
        description=description,
        equipment=equipment_list
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return RedirectResponse(url="/documents/", status_code=303)

@app.get("/documents/{document_id}", response_class=HTMLResponse)
async def read_document_detail(request: Request, document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    # 多对多关系，直接使用document.equipment访问设备列表
    equipment_list = document.equipment
    return templates.TemplateResponse("document_detail.html", {"request": request, "document": document, "equipment": equipment_list})

@app.post("/documents/delete/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    # 删除文件
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    # 删除数据库记录
    db.delete(document)
    db.commit()
    return RedirectResponse(url="/documents/", status_code=303)

@app.get("/documents/{document_id}/edit", response_class=HTMLResponse)
async def edit_document_form(request: Request, document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    equipment_list = db.query(EquipmentDB).all()
    return templates.TemplateResponse("document_form.html", {"request": request, "document": document, "equipment_list": equipment_list})

@app.post("/documents/{document_id}/edit")
async def update_document(
    request: Request,
    document_id: int,
    title: str = Form(...),
    category: str = Form(...),
    description: str | None = Form(None),
    equipment_ids: list[int] = Form(None),
    db: Session = Depends(get_db)
):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    # 处理equipment_ids列表
    valid_equipment_ids = []
    if equipment_ids:
        for eq_id in equipment_ids:
            try:
                if eq_id and str(eq_id).strip():
                    valid_equipment_ids.append(int(eq_id))
            except ValueError:
                continue
    # 获取关联的设备对象
    equipment_list = db.query(EquipmentDB).filter(EquipmentDB.id.in_(valid_equipment_ids)).all()
    
    # 更新文档信息
    document.title = title
    document.category = category
    document.description = description
    document.equipment = equipment_list
    
    db.commit()
    db.refresh(document)
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)

# 文档查看路由
@app.get("/documents/view/{document_id}")
async def view_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    # 检查文件路径和存在性
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 确保文件可读
    try:
        with open(document.file_path, 'rb') as f:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件读取失败: {str(e)}")
    
    if document.file_type.startswith("application/pdf") or document.file_type.startswith("image/"):
        return FileResponse(document.file_path, media_type=document.file_type)
    else:
        raise HTTPException(status_code=400, detail="不支持的文件类型，无法直接查看")

@app.get("/monitoring/", response_class=HTMLResponse)
async def equipment_monitoring(context = Depends(get_template_context)):
    return templates.TemplateResponse("equipment_monitoring.html", context)

# 设备类型维护周期映射表 (月)
MAINTENANCE_CYCLE_MAP = {
    'DCS': 60,
    'PLC': 12,
    '变送器': 1,
    '热电阻': 12,
    '热电偶': 24,
    '调节阀': 3,
    '切断阀': 6,
    '物位计': 3,
    '压力表': 3,
    '双金属温度计': 6,
    '分析仪表': 1,
    '流量计': 24,
    '其它': 1
}

@app.get("/monitoring/data/")
async def equipment_monitoring_data(db: Session = Depends(get_db)):
    # 获取所有设备
    equipments = db.query(EquipmentDB).all()
    
    # 获取所有维护记录
    maintenances = db.query(MaintenanceDB).all()
    
    # 准备设备详情数据
    equipment_details = []
    for eq in equipments:
        # 计算已使用时间(月)
        if eq.installation_date:
            used_months = (datetime.now() - eq.installation_date).days / 30.0
        else:
            used_months = 0
        
        # 获取工作寿命（如果没有设置，则使用默认值）
        working_life = eq.working_life if eq.working_life is not None else WORKING_LIFE_MAP.get(eq.equipment_type, 60)
        
        # 计算剩余寿命(月)
        remaining_life = working_life - used_months
        
        # 获取维护周期
        maintenance_cycle = MAINTENANCE_CYCLE_MAP.get(eq.equipment_type, 0)
        
        equipment_details.append({
            "id": eq.id,
            "name": eq.name,
            "model": eq.model,
            "type": eq.equipment_type,
            "working_life": working_life,
            "maintenance_cycle": maintenance_cycle,
            "used_months": used_months,
            "remaining_life": remaining_life,
            "next_maintenance": eq.next_maintenance.strftime('%Y-%m-%d') if eq.next_maintenance else None,
            "status": eq.status
        })
    
    # 准备维护历史数据（从本月开始未来6个月）
    maintenance_history = []
    today = date.today()
    for i in range(6):
        # 计算未来月份
        if today.month + i > 12:
            month_date = date(today.year + (today.month + i) // 12, (today.month + i) % 12, 1)
        else:
            month_date = date(today.year, today.month + i, 1)
        
        # 计算下一个月
        if month_date.month < 12:
            next_month = date(month_date.year, month_date.month + 1, 1)
        else:
            next_month = date(month_date.year + 1, 1, 1)
        
        # 对于未来的月份，维护次数为0（因为还未发生）
        # 对于当前和过去的月份，从数据库查询
        if month_date <= today:
            count = len([m for m in maintenances if month_date <= m.maintenance_date.date() < next_month])
        else:
            count = 0
        
        month_name = month_date.strftime("%Y-%m")
        maintenance_history.append({"month": month_name, "count": count})
    
    # 按月份排序（虽然已经是按顺序生成的，但保留排序逻辑确保正确性）
    maintenance_history.sort(key=lambda x: x["month"])
    
    # 设备类型分布
    equipment_type_distribution = {}
    for eq in equipments:
        eq_type = eq.equipment_type or "未分类"
        equipment_type_distribution[eq_type] = equipment_type_distribution.get(eq_type, 0) + 1
    
    return JSONResponse(content={
        "equipment_details": equipment_details,
        "maintenance_history": maintenance_history,
        "equipment_type_distribution": equipment_type_distribution
    })

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 配件管理相关路由
@app.get("/spare-parts/locations/", response_class=HTMLResponse)
async def list_locations(request: Request, db: Session = Depends(get_db)):
    locations = db.query(LocationDB).all()
    return templates.TemplateResponse("location_list.html", {"request": request, "locations": locations})

@app.get("/spare-parts/locations/create/", response_class=HTMLResponse)
async def create_location_form(request: Request):
    return templates.TemplateResponse("location_form.html", {"request": request})

@app.post("/spare-parts/locations/create/")
async def create_location(
    request: Request,
    name: str = Form(...),
    code: str = Form(...),
    category: str = Form(None),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    db_location = LocationDB(
        name=name,
        code=code,
        category=category,
        description=description
    )
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return RedirectResponse(url="/spare-parts/locations/", status_code=303)

@app.get("/spare-parts/locations/edit/{location_id}", response_class=HTMLResponse)
async def edit_location_form(request: Request, location_id: int, db: Session = Depends(get_db)):
    location = db.query(LocationDB).filter(LocationDB.id == location_id).first()
    if location is None:
        raise HTTPException(status_code=404, detail="库位未找到")
    return templates.TemplateResponse("location_form.html", {"request": request, "location": location, "edit": True})

@app.post("/spare-parts/locations/edit/{location_id}")
async def update_location(
    request: Request,
    location_id: int,
    name: str = Form(...),
    code: str = Form(...),
    category: str = Form(None),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    location = db.query(LocationDB).filter(LocationDB.id == location_id).first()
    if location is None:
        raise HTTPException(status_code=404, detail="库位未找到")
    
    location.name = name
    location.code = code
    location.category = category
    location.description = description
    location.updated_at = datetime.now()
    
    db.commit()
    return RedirectResponse(url="/spare-parts/locations/", status_code=303)

@app.get("/spare-parts/locations/delete/{location_id}")
async def delete_location(request: Request, location_id: int, db: Session = Depends(get_db)):
    location = db.query(LocationDB).filter(LocationDB.id == location_id).first()
    if location is None:
        raise HTTPException(status_code=404, detail="库位未找到")
    
    # 检查是否有配件关联到此库位
    has_spare_parts = db.query(SparePartDB).filter(SparePartDB.location_id == location_id).first() is not None
    if has_spare_parts:
        raise HTTPException(status_code=400, detail="此库位关联有配件，无法删除")
    
    # 检查是否有库存交易关联到此库位
    has_inventory = db.query(InventoryDB).filter(InventoryDB.location_id == location_id).first() is not None
    if has_inventory:
        raise HTTPException(status_code=400, detail="此库位有关联的库存记录，无法删除")
    
    db.delete(location)
    db.commit()
    return RedirectResponse(url="/spare-parts/locations/", status_code=303)

@app.get("/spare-parts/parts/", response_class=HTMLResponse)
async def list_spare_parts(context = Depends(get_template_context), db: Session = Depends(get_db)):
    parts = db.query(SparePartDB).options(joinedload(SparePartDB.location)).all()
    locations = db.query(LocationDB).all()
    context.update({"parts": parts, "locations": locations})
    return templates.TemplateResponse("spare_part_list.html", context)

@app.get("/spare-parts/parts/create/", response_class=HTMLResponse)
async def create_spare_part_form(request: Request, db: Session = Depends(get_db)):
    locations = db.query(LocationDB).all()
    return templates.TemplateResponse("spare_part_form.html", {"request": request, "locations": locations})

@app.post("/spare-parts/parts/create/")
async def create_spare_part(
    request: Request,
    name: str = Form(...),
    code: str = Form(...),
    specification: str = Form(None),
    unit: str = Form(...),
    location_id: int = Form(None),
    safety_stock: int = Form(0),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    db_part = SparePartDB(
        name=name,
        code=code,
        specification=specification,
        unit=unit,
        location_id=location_id,
        safety_stock=safety_stock,
        description=description
    )
    db.add(db_part)
    db.commit()
    db.refresh(db_part)
    return RedirectResponse(url="/spare-parts/parts/", status_code=303)

# 故障报告路由
@app.get("/fault-reports/", response_class=HTMLResponse)
async def fault_report_list(request: Request, db: Session = Depends(get_db), page: int = 1):
    # 分页设置
    per_page = 10
    offset = (page - 1) * per_page

    # 查询故障报告
    reports = db.query(FaultReportDB).offset(offset).limit(per_page).all()
    total_reports = db.query(FaultReportDB).count()
    total_pages = (total_reports + per_page - 1) // per_page

    # 统计数据
    pending_reports = db.query(FaultReportDB).filter(FaultReportDB.status == "pending").count()
    resolved_reports = db.query(FaultReportDB).filter(FaultReportDB.status == "resolved").count()
    hardware_faults = db.query(FaultReportDB).filter(FaultReportDB.fault_type.contains("硬件")).count()
    software_faults = db.query(FaultReportDB).filter(FaultReportDB.fault_type.contains("软件")).count()

    # 故障类型分布
    fault_types = ["硬件故障", "软件故障", "连接故障", "配置故障", "其他故障"]
    fault_type_counts = []
    for ft in fault_types:
        count = db.query(FaultReportDB).filter(FaultReportDB.fault_type == ft).count()
        fault_type_counts.append(count)

    return templates.TemplateResponse("fault_report_list.html", {
        "request": request,
        "reports": reports,
        "total_reports": total_reports,
        "pending_reports": pending_reports,
        "resolved_reports": resolved_reports,
        "hardware_faults": hardware_faults,
        "software_faults": software_faults,
        "fault_types": fault_types,
        "fault_type_counts": fault_type_counts,
        "current_page": page,
        "total_pages": total_pages
    })

@app.get("/fault-reports/create/", response_class=HTMLResponse)
async def create_fault_report_form(request: Request, db: Session = Depends(get_db)):
    # 获取所有设备
    equipments = db.query(EquipmentDB).all()
    return templates.TemplateResponse("fault_report_form.html", {
        "request": request,
        "equipments": equipments,
        "report": None,
        "datetime": datetime
    })

@app.post("/fault-reports/create/")
async def create_fault_report(request: Request, db: Session = Depends(get_db)):
    form_data = await request.form()

    # 处理文件上传
    image_path = None
    if "image" in form_data and form_data["image"].file:
        image_file = form_data["image"]
        # 确保上传目录存在
        upload_dir = os.path.join(os.path.dirname(__file__), "static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        # 生成唯一文件名
        filename = f"{uuid.uuid4()}_{image_file.filename}"
        image_path = os.path.join(upload_dir, filename)
        # 保存文件
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        # 保存相对路径
        image_path = f"/static/uploads/{filename}"

    # 创建故障报告
    report = FaultReportDB(
        equipment_id=int(form_data["equipment_id"]),
        report_date=datetime.strptime(form_data["report_date"], "%Y-%m-%d"),
        fault_type=form_data["fault_type"],
        description=form_data["description"],
        created_by="****",  # 默认值
        image_path=image_path
    )

    db.add(report)
    db.commit()
    db.refresh(report)

    return RedirectResponse(url="/fault-reports/", status_code=303)

@app.get("/fault-reports/{report_id}", response_class=HTMLResponse)
async def view_fault_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    report = db.query(FaultReportDB).filter(FaultReportDB.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="故障报告未找到")
    # 安全获取设备名称
    equipment_name = report.equipment.name if report.equipment else "未知设备"
    return templates.TemplateResponse("fault_report_detail.html", {
        "request": request,
        "report": report
    })

@app.get("/fault-reports/{report_id}/edit/", response_class=HTMLResponse)
async def edit_fault_report_form(report_id: int, request: Request, db: Session = Depends(get_db)):
    report = db.query(FaultReportDB).filter(FaultReportDB.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="故障报告未找到")
    equipments = db.query(EquipmentDB).all()
    return templates.TemplateResponse("fault_report_form.html", {
        "request": request,
        "report": report,
        "equipments": equipments
    })

@app.post("/fault-reports/{report_id}/edit/")
async def update_fault_report(report_id: int, request: Request, db: Session = Depends(get_db)):
    report = db.query(FaultReportDB).filter(FaultReportDB.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="故障报告未找到")

    form_data = await request.form()

    # 更新报告信息
    report.equipment_id = int(form_data["equipment_id"])
    report.report_date = datetime.strptime(form_data["report_date"], "%Y-%m-%d")
    report.fault_type = form_data["fault_type"]
    report.description = form_data["description"]
    report.status = form_data["status"]

    # 处理状态更新
    if form_data["status"] == "resolved" and not report.resolved_date:
        report.resolved_date = datetime.now()
        report.resolution = form_data.get("resolution", "")
        report.cost = float(form_data.get("cost", 0)) if form_data.get("cost") else None
        report.duration = float(form_data.get("duration", 0)) if form_data.get("duration") else None

    db.commit()
    db.refresh(report)

    return RedirectResponse(url=f"/fault-reports/{report_id}", status_code=303)

@app.post("/fault-reports/delete/{report_id}")
async def delete_fault_report(report_id: int, db: Session = Depends(get_db)):
    report = db.query(FaultReportDB).filter(FaultReportDB.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="故障报告未找到")

    db.delete(report)
    db.commit()

    return JSONResponse(content={"status": "success"})

# 检维修管理相关模型
class InspectionStandardDB(Base):
    __tablename__ = "inspection_standards"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    equipment_type = Column(String(100), nullable=False)
    cycle = Column(Integer, nullable=False)
    cycle_unit = Column(String(20), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    items = relationship("InspectionStandardItemDB", back_populates="standard_rel", cascade="all, delete-orphan")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class InspectionStandardItemDB(Base):
    __tablename__ = "inspection_standard_items"
    id = Column(Integer, primary_key=True, index=True)
    standard_id = Column(Integer, ForeignKey("inspection_standards.id"), nullable=False)
    name = Column(String(255), nullable=False)
    standard = Column(String(255), nullable=False)
    method = Column(String(255), nullable=False)
    standard_rel = relationship("InspectionStandardDB", back_populates="items")


# 检维修管理路由
@app.get("/maintenance/", response_class=HTMLResponse)
async def maintenance_home(context = Depends(get_template_context), db: Session = Depends(get_db)):
    # 获取故障报告总数
    total_fault_reports = db.query(FaultReportDB).count()
    context.update({
        "fault_reports": total_fault_reports
    })
    return templates.TemplateResponse("maintenance.html", context)

@app.get("/maintenance/statistics/", response_class=HTMLResponse)
async def maintenance_statistics(request: Request, date: str = None, db: Session = Depends(get_db)):
    # 构建查询基础
    query = db.query(MaintenanceDB)
    
    # 如果提供了日期参数，按日期过滤
    selected_date = None
    if date:
        try:
            selected_date = datetime.strptime(date, '%Y-%m-%d').date()
            query = query.filter(MaintenanceDB.maintenance_date == selected_date)
        except ValueError:
            # 日期格式错误，不进行过滤
            pass
    
    # 获取统计数据
    total_maintenances = query.count()
    
    # 按设备类型统计
    equipment_query = db.query(
        EquipmentDB.name,
        func.count(MaintenanceDB.id).label('count')
    ).join(
        MaintenanceDB, EquipmentDB.id == MaintenanceDB.equipment_id
    )
    
    # 如果有选择的日期，过滤设备统计
    if selected_date:
        equipment_query = equipment_query.filter(MaintenanceDB.maintenance_date == selected_date)
    
    maintenance_by_equipment = equipment_query.group_by(
        EquipmentDB.name
    ).all()
    
    # 按维护类型统计
    type_query = db.query(
        MaintenanceDB.maintenance_type,
        func.count(MaintenanceDB.id).label('count')
    )
    
    # 如果有选择的日期，过滤类型统计
    if selected_date:
        type_query = type_query.filter(MaintenanceDB.maintenance_date == selected_date)
    
    maintenance_by_type = type_query.group_by(
        MaintenanceDB.maintenance_type
    ).all()
    
    # 获取维护详情
    maintenance_details = []
    maintenances_query = db.query(MaintenanceDB).order_by(MaintenanceDB.maintenance_date.desc())
    
    if selected_date:
        maintenances_query = maintenances_query.filter(MaintenanceDB.maintenance_date == selected_date)
    else:
        maintenances_query = maintenances_query.limit(10)
    
    maintenances = maintenances_query.all()
    for maintenance in maintenances:
        # 获取设备名称
        equipment = db.query(EquipmentDB).filter(EquipmentDB.id == maintenance.equipment_id).first()
        equipment_name = equipment.name if equipment else "未知设备"
        
        maintenance_details.append({
            'equipment': equipment_name,
            'type': maintenance.maintenance_type,
            'date': maintenance.maintenance_date.strftime('%Y-%m-%d'),
            'person': maintenance.maintenance_person or "未知",
        })
    
    return templates.TemplateResponse("maintenance_statistics.html", {
        "request": request,
        "total_maintenances": total_maintenances,
        "maintenance_by_equipment": maintenance_by_equipment,
        "maintenance_by_type": maintenance_by_type,
        "maintenance_details": maintenance_details,
        "selected_date": selected_date.strftime('%Y-%m-%d') if selected_date else None
    })

@app.get("/maintenance/history/", response_class=HTMLResponse)
async def maintenance_history(request: Request, db: Session = Depends(get_db), page: int = 1):
    # 分页设置
    per_page = 10
    offset = (page - 1) * per_page

    # 查询维护历史记录
    maintenances = db.query(MaintenanceDB).offset(offset).limit(per_page).all()
    total_maintenances = db.query(MaintenanceDB).count()
    total_pages = (total_maintenances + per_page - 1) // per_page

    return templates.TemplateResponse("maintenance_history.html", {
        "request": request,
        "maintenances": maintenances,
        "total_maintenances": total_maintenances,
        "current_page": page,
        "total_pages": total_pages
    })

@app.get("/inspection/standards/", response_class=HTMLResponse)
async def read_inspection_standards(request: Request, db: Session = Depends(get_db)):
    standards = db.query(InspectionStandardDB).all()
    return templates.TemplateResponse("inspection_standard.html", {"request": request, "standards": standards})

@app.get("/inspection/standards/create/", response_class=HTMLResponse)
async def create_inspection_standard_form(request: Request):
    return templates.TemplateResponse("inspection_standard_form.html", {"request": request})

@app.post("/inspection/standards/create/")
async def create_inspection_standard(
    request: Request,
    db: Session = Depends(get_db)
):
    # 直接从request获取所有表单数据
    form_data = await request.form()
    
    # 提取基本字段
    name = form_data.get("name")
    equipment_type = form_data.get("equipment_type")
    cycle = form_data.get("cycle")
    cycle_unit = form_data.get("cycle_unit")
    description = form_data.get("description")
    
    # 提取带[]后缀的数组字段
    item_name = form_data.getlist("item_name[]")
    item_standard = form_data.getlist("item_standard[]")
    item_method = form_data.getlist("item_method[]")
    
    # 验证基本字段
    if not name or not equipment_type or not cycle or not cycle_unit:
        raise HTTPException(
            status_code=422,
            detail="基本信息不完整，请确保填写了所有必填字段"
        )
    
    # 转换cycle为整数
    try:
        cycle = int(cycle)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="巡检周期必须是数字"
        )
    
    # 验证数组字段
    if not item_name or not item_standard or not item_method:
        raise HTTPException(
            status_code=422,
            detail="巡检项目信息不完整，请至少添加一个巡检项目"
        )
        
    # 确保数组长度一致
    if not (len(item_name) == len(item_standard) == len(item_method)):
        raise HTTPException(
            status_code=422,
            detail="巡检项目字段数量不匹配，请重新填写表单"
        )
        
    # 验证每个巡检项目的字段不为空
    for i in range(len(item_name)):
        if not item_name[i].strip() or not item_standard[i].strip() or not item_method[i].strip():
            raise HTTPException(
                status_code=422,
                detail=f"第{i+1}个巡检项目的信息不完整，请填写所有必填字段"
            )

    # 创建巡检标准
    db_standard = InspectionStandardDB(
        name=name,
        equipment_type=equipment_type,
        cycle=cycle,
        cycle_unit=cycle_unit,
        description=description
    )
    db.add(db_standard)
    db.commit()
    db.refresh(db_standard)

    # 创建巡检项目
    for name_val, standard_val, method_val in zip(item_name, item_standard, item_method):
        db_item = InspectionStandardItemDB(
            standard_id=db_standard.id,
            name=name_val,
            standard=standard_val,
            method=method_val
        )
        db.add(db_item)
    db.commit()

    return RedirectResponse(url="/inspection/standards/", status_code=303)

# 导入备份模块并在后台线程中启动自动备份
import threading
from backup_database import backup_database, start_auto_backup

# 只执行一次备份，不启动调度器
backup_database()

# 可选：如果需要定期备份，可以在后台线程中运行
# def run_backup_in_background():
#     start_auto_backup()
# 
# backup_thread = threading.Thread(target=run_backup_in_background, daemon=True)
# backup_thread.start()

# 运行应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5173)

from sqlalchemy.orm import joinedload

@app.get("/inspection/standards/{standard_id}/edit/", response_class=HTMLResponse)
async def edit_inspection_standard_form(request: Request, standard_id: int, db: Session = Depends(get_db)):
    standard = db.query(InspectionStandardDB).options(joinedload(InspectionStandardDB.items)).filter(InspectionStandardDB.id == standard_id).first()
    if standard is None:
        raise HTTPException(status_code=404, detail="点巡检标准未找到")
    return templates.TemplateResponse("inspection_standard_form.html", {"request": request, "standard": standard})

@app.post("/inspection/standards/{standard_id}/edit/")
async def update_inspection_standard(
    request: Request,
    standard_id: int,
    db: Session = Depends(get_db)
):
    # 直接从request获取所有表单数据
    form_data = await request.form()
    
    # 提取基本字段
    name = form_data.get("name")
    equipment_type = form_data.get("equipment_type")
    cycle = form_data.get("cycle")
    cycle_unit = form_data.get("cycle_unit")
    description = form_data.get("description")
    
    # 提取带[]后缀的数组字段
    item_name = form_data.getlist("item_name[]")
    item_standard = form_data.getlist("item_standard[]")
    item_method = form_data.getlist("item_method[]")
    
    # 验证基本字段
    if not name or not equipment_type or not cycle or not cycle_unit:
        raise HTTPException(
            status_code=422,
            detail="基本信息不完整，请确保填写了所有必填字段"
        )
    
    # 转换cycle为整数
    try:
        cycle = int(cycle)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="巡检周期必须是数字"
        )
    
    # 验证数组字段
    if not item_name or not item_standard or not item_method:
        raise HTTPException(
            status_code=422,
            detail="巡检项目信息不完整，请至少添加一个巡检项目"
        )
        
    # 确保数组长度一致
    if not (len(item_name) == len(item_standard) == len(item_method)):
        raise HTTPException(
            status_code=422,
            detail="巡检项目字段数量不匹配，请重新填写表单"
        )
        
    # 验证每个巡检项目的字段不为空
    for i in range(len(item_name)):
        if not item_name[i].strip() or not item_standard[i].strip() or not item_method[i].strip():
            raise HTTPException(
                status_code=422,
                detail=f"第{i+1}个巡检项目的信息不完整，请填写所有必填字段"
            )
    
    # 查找要更新的标准
    standard = db.query(InspectionStandardDB).filter(InspectionStandardDB.id == standard_id).first()
    if standard is None:
        raise HTTPException(status_code=404, detail="点巡检标准未找到")

    # 更新标准信息
    standard.name = name
    standard.equipment_type = equipment_type
    standard.cycle = cycle
    standard.cycle_unit = cycle_unit
    standard.description = description
    standard.updated_at = datetime.now()

    # 先删除现有项目
    db.query(InspectionStandardItemDB).filter(InspectionStandardItemDB.standard_id == standard_id).delete()

    # 添加新项目
    for name, standard_val, method in zip(item_name, item_standard, item_method):
        db_item = InspectionStandardItemDB(
            standard_id=standard_id,
            name=name,
            standard=standard_val,
            method=method
        )
        db.add(db_item)

    db.commit()
    return RedirectResponse(url="/inspection/standards/", status_code=303)

@app.get("/inspection/standards/{standard_id}/detail/", response_class=HTMLResponse)
async def detail_inspection_standard(request: Request, standard_id: int, db: Session = Depends(get_db)):
    standard = db.query(InspectionStandardDB).filter(InspectionStandardDB.id == standard_id).first()
    if standard is None:
        raise HTTPException(status_code=404, detail="点巡检标准未找到")
    # 查询相关的巡检项目
    items = db.query(InspectionStandardItemDB).filter(InspectionStandardItemDB.standard_id == standard_id).all()
    return templates.TemplateResponse("inspection_standard_detail.html", {"request": request, "standard": standard, "items": items})

@app.post("/inspection/standards/delete/{standard_id}")
async def delete_inspection_standard(standard_id: int, db: Session = Depends(get_db)):
    standard = db.query(InspectionStandardDB).filter(InspectionStandardDB.id == standard_id).first()
    if standard is None:
        raise HTTPException(status_code=404, detail="点巡检标准未找到")
    db.delete(standard)
    db.commit()
    return RedirectResponse(url="/inspection/standards/", status_code=303)

# 用户管理相关路由

# 登录页面
# 用户认证相关路由

# 登录页面
@app.get("/login/", response_class=HTMLResponse)
async def login_form(request: Request, db: Session = Depends(get_db)):
    current_user = await get_current_user(request, db)
    if current_user:
        return RedirectResponse(url="/", status_code=307)
    return templates.TemplateResponse("new_login.html", {"request": request})

# 登录处理
@app.post("/login/")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(None),
    db: Session = Depends(get_db)
):
    # 检查用户是否存在
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user or not user.is_active:
        return templates.TemplateResponse(
            "new_login.html", 
            {"request": request, "error_message": "用户名不存在"}
        )
    
    # 游客用户无需密码登录
    if username == "guest":
        pass
    # 其他用户需要验证密码
    elif not password or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "new_login.html", 
            {"request": request, "error_message": "密码错误"}
        )
    
    # 更新最后登录时间
    user.last_login = datetime.now()
    db.commit()
    
    # 创建访问令牌
    access_token = create_access_token(data={"sub": user.username})
    
    # 设置Cookie并重定向
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return response

# 登出
@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login/", status_code=303)
    response.delete_cookie("access_token")
    return response