import uvicorn
from fastapi.routing import APIRoute
from main import app

# 打印所有注册的路由，特别关注登录相关的路由
def list_routes():
    routes = app.routes
    print("注册的路由列表:")
    login_routes = []
    for route in routes:
        if isinstance(route, APIRoute):
            if "login" in route.path.lower():
                login_routes.append(f"路径: {route.path}, 方法: {route.methods}")
    
    if login_routes:
        print("\n登录相关路由:")
        for route in login_routes:
            print(route)
    else:
        print("\n未找到登录相关路由!")

if __name__ == "__main__":
    list_routes()
    # 启动应用
    uvicorn.run(app, host="0.0.0.0", port=5173)