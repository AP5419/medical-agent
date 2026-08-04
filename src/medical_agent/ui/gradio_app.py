# -*- coding: utf-8 -*-
# Gradio Web UI - 医疗智能体交互界面
import json

import gradio as gr
import httpx

BACKEND_URL = "http://localhost:8080"

css = """
.medical-footer {
    text-align: center;
    color: #888;
    font-size: 12px;
    margin-top: 20px;
}
.login-panel {
    max-width: 400px;
    margin: 0 auto;
}
"""

# 角色对应的快速操作按钮标签
ROLE_ACTIONS = {
    "patient": [
        "预约问诊", "查看报告", "药品查询",
        "智能问诊", "健康知识",
    ],
    "doctor": [
        "辅助诊断", "报告解读", "开具处方",
        "药品查询", "临床指南", "手术数据",
    ],
    "pharmacist": [
        "处方审核", "药品查询", "报告解读",
        "药物相互作用", "处方点评",
    ],
    "admin": [
        "用户管理", "审计日志", "系统配置",
        "数据统计", "知识库管理",
    ],
}

# 当前会话状态
current_token = None
current_user_info = None
current_session_id = None


def _make_auth_headers():
    """构建带认证信息的请求头"""
    if current_token:
        return {"Authorization": f"Bearer {current_token}"}
    return {}


async def login(username: str, password: str, role: str):
    """用户登录"""
    global current_token, current_user_info, current_session_id

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BACKEND_URL}/api/v1/auth/login",
                json={"username": username, "password": password},
                timeout=10.0,
            )
            if resp.status_code != 200:
                detail = resp.json().get("detail", "登录失败")
                return f"❌ {detail}", "", gr.Column(visible=False), gr.Column(visible=True)
            data = resp.json()
            current_token = data["access_token"]
            current_user_info = data["user"]
            current_session_id = _generate_session_id()

            user_info_text = (
                f"✅ 登录成功\n"
                f"身份: {current_user_info['role']} | 用户: {current_user_info['real_name'] or current_user_info['username']}"
            )
            return (
                user_info_text,
                f"欢迎, {current_user_info['real_name'] or current_user_info['username']}!",
                gr.Column(visible=True),
                gr.Column(visible=False),
            )
        except Exception as e:
            return f"❌ 连接失败: {str(e)}", "", gr.Column(visible=False), gr.Column(visible=True)


async def register(username: str, password: str, role: str, real_name: str):
    """用户注册"""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BACKEND_URL}/api/v1/auth/register",
                json={
                    "username": username,
                    "password": password,
                    "role": role,
                    "real_name": real_name,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                return f"✅ 注册成功！用户 {username}（{role}）已创建，请切换到登录页面登录。"
            elif resp.status_code == 409:
                return f"❌ 用户名 {username} 已存在，请更换用户名。"
            else:
                detail = resp.json().get("detail", "注册失败")
                return f"❌ {detail}"
        except Exception as e:
            return f"❌ 连接失败: {str(e)}"


def logout():
    """用户登出"""
    global current_token, current_user_info, current_session_id
    current_token = None
    current_user_info = None
    current_session_id = None
    return "", "", gr.Column(visible=False), gr.Column(visible=True)


def _generate_session_id():
    """生成会话ID"""
    import uuid
    return uuid.uuid4().hex


async def chat_send(message: str, history: list):
    """发送流式对话消息"""
    global current_token, current_user_info, current_session_id

    if not current_token or not current_user_info:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ 请先登录"})
        yield history
        return

    if not message or not message.strip():
        return

    if current_session_id is None:
        current_session_id = _generate_session_id()

    # 添加用户消息到历史
    history.append({"role": "user", "content": message})
    # 占位 assistant 消息（流式更新时替换）
    history.append({"role": "assistant", "content": ""})

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/v1/chat/stream",
                json={
                    "user_id": str(current_user_info["id"]),
                    "session_id": current_session_id,
                    "message": message,
                },
                headers=_make_auth_headers(),
            ) as response:
                if response.status_code != 200:
                    history[-1] = {"role": "assistant", "content": f"❌ 请求失败 ({response.status_code})"}
                    yield history
                    return

                full_response = ""
                current_pos = len(history) - 1

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        if chunk.get("event") == "token":
                            full_response += chunk.get("data", "")
                            history[current_pos] = {"role": "assistant", "content": full_response}
                            yield history
                        elif chunk.get("event") == "done":
                            break
                        elif chunk.get("event") == "error":
                            history[current_pos] = {"role": "assistant", "content": f"❌ {chunk.get('data', '')}"}
                            yield history
                            return
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            history[-1] = {"role": "assistant", "content": f"❌ 连接失败: {str(e)}"}
            yield history


async def quick_action(action: str):
    """处理快速操作按钮点击"""
    action_messages = {
        "预约问诊": "请帮我预约一次在线问诊",
        "查看报告": "请帮我查看最近的检查报告",
        "药品查询": "请帮我查询药品信息",
        "智能问诊": "请描述您的症状，我会为您分析",
        "健康知识": "请分享一些健康知识",
        "辅助诊断": "请辅助进行诊断分析",
        "报告解读": "请解读这份医疗报告",
        "开具处方": "请根据诊断开具处方",
        "临床指南": "请查询相关临床指南",
        "手术数据": "请查询相关手术数据",
        "用户管理": "请打开用户管理面板",
        "审计日志": "请查看系统审计日志",
        "系统配置": "请查看系统配置信息",
        "数据统计": "请查看系统数据统计",
        "知识库管理": "请管理知识库内容",
    }
    return action_messages.get(action, action)


def build_ui():
    """构建Gradio界面"""
    with gr.Blocks(title="灵枢医疗智能体系统") as demo:
        gr.Markdown("# 🏥 灵枢医疗多智能体系统")

        # 当前状态存储
        login_state = gr.State("")

        with gr.Column(visible=True, elem_classes="login-panel") as login_panel:
            with gr.Tabs() as auth_tabs:
                # ── 登录标签页 ──
                with gr.Tab("登录"):
                    login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                    login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                    login_role = gr.Dropdown(
                        choices=["patient", "doctor", "pharmacist", "admin"],
                        value="patient",
                        label="角色",
                    )
                    login_button = gr.Button("登录", variant="primary")
                    login_result = gr.Markdown("")

                # ── 注册标签页 ──
                with gr.Tab("注册"):
                    gr.Markdown("### 新建账号")
                    reg_username = gr.Textbox(label="用户名", placeholder="3-50个字符")
                    reg_password = gr.Textbox(label="密码", type="password", placeholder="至少6位")
                    reg_role = gr.Dropdown(
                        choices=["patient", "doctor", "pharmacist", "admin"],
                        value="patient",
                        label="角色",
                    )
                    reg_realname = gr.Textbox(label="真实姓名（可选）", placeholder="用于展示")
                    register_button = gr.Button("注册", variant="secondary")
                    register_result = gr.Markdown("")

        with gr.Column(visible=False) as chat_panel:
            # 头部信息
            gr.Markdown("## 智能问诊对话")
            welcome_msg = gr.Markdown("")

            # 快速操作按钮
            with gr.Row() as quick_actions_row:
                action_buttons = []
                for i in range(5):
                    btn = gr.Button(f"操作{i + 1}", visible=False)
                    action_buttons.append(btn)

            # 聊天区域
            chatbot = gr.Chatbot(label="对话记录", height=450)
            msg_input = gr.Textbox(
                label="输入消息",
                placeholder="请输入您的医疗问题...",
                lines=2,
            )
            with gr.Row():
                send_button = gr.Button("发送", variant="primary")
                clear_button = gr.Button("清空对话")
            logout_button = gr.Button("退出登录")

        # 底部免责声明
        gr.Markdown(
            """
            ---
            *⚠️ 医疗免责声明：本系统提供的信息仅供参考，不构成医疗诊断或治疗建议。*
            *如有身体不适，请及时就医。请勿依据本系统内容自行诊疗。*
            """,
            elem_classes="medical-footer",
        )

        # --- 事件绑定 ---

        # 登录
        login_button.click(
            fn=login,
            inputs=[login_username, login_password, login_role],
            outputs=[login_result, welcome_msg, chat_panel, login_panel],
        )

        # 注册
        register_button.click(
            fn=register,
            inputs=[reg_username, reg_password, reg_role, reg_realname],
            outputs=[register_result],
        )

        # 退出登录
        logout_button.click(
            fn=logout,
            inputs=[],
            outputs=[login_result, welcome_msg, chat_panel, login_panel],
        ).then(
            fn=lambda: ([],),
            outputs=[chatbot],
        )

        # 发送消息（流式）
        send_button.click(
            fn=chat_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        )

        # 回车发送
        msg_input.submit(
            fn=chat_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        )

        # 清空对话
        clear_button.click(
            fn=lambda: [],
            inputs=[],
            outputs=[chatbot],
        )

        # 快速操作按钮 - 更新角色对应的操作按钮
        def update_action_buttons(role):
            """根据角色更新快捷操作按钮"""
            actions = ROLE_ACTIONS.get(role, [])
            updates = []
            for i in range(5):
                if i < len(actions):
                    updates.append(gr.update(value=actions[i], visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        # 快速操作按钮点击 - 填充输入框
        for idx, btn in enumerate(action_buttons):
            btn.click(
                fn=quick_action,
                inputs=[btn],
                outputs=[msg_input],
            )

        # 登录成功后更新快速操作按钮
        login_result.change(
            fn=lambda info: (
                update_action_buttons(current_user_info.get("role", ""))
                if (info and "✅" in info and current_user_info)
                else [gr.update(visible=False)] * 5
            ),
            inputs=[login_result],
            outputs=action_buttons,
        )

    return demo


def launch_ui():
    """启动Gradio Web UI"""
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, css=css, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_ui()
