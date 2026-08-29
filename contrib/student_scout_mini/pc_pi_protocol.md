# PC-Pi 控制器通信协议说明

## 概述

本文档描述 PC (Windows) 与树莓派 (Raspberry Pi 5) 之间的实时控制通信协议，用于 MPPI 动态避障实车测试。

### 架构

```
PC (Windows) --TCP:57720--> Pi 网关 --CAN:can0--> Scout Mini 底盘
     |                              |
  simple_pc_controller.py    minimal_gateway4.py
  (PC端MPPI控制器)           (Pi端简化网关)
```

### 当前状态 (2026-08-29)

- ✅ CAN总线正常工作 (test_move2.py 验证通过)
- ✅ PC-Pi TCP 通信已打通 (HELLO/STATUS/COMMAND帧)
- ⏳ 完整MPPI控制器待集成到网关
- 📍 网关监听端口: 57720 (控制) / 57721 (扫描，可选)

---

## 帧格式

### 协议头

```
+--------+--------+----------+
| MAGIC  | TYPE   | LENGTH   |
| 4字节  | 1字节  | 4字节    |
+--------+--------+----------+
```

- **MAGIC**: `b'RMP1'` (固定值)
- **TYPE**: 帧类型 (1字节)
- **LENGTH**: payload 长度 (4字节，网络字节序)

### 帧类型

| 类型名 | 值 | 说明 |
|--------|------|------|
| FRAME_HELLO | 4 | 握手帧，首次连接发送 |
| FRAME_STATUS | 2 | 状态帧，网关主动发送 |
| FRAME_COMMAND | 3 | 命令帧，PC发送控制指令 |
| FRAME_STOP | 5 | 停止帧，终止通信 |
| FRAME_SCAN | 1 | 雷达扫描帧 (可选) |

---

## HELLO 帧 (握手)

### PC -> Pi

```json
{
  "token": "pc_test_token",
  "scan_udp_port": 57701
}
```

- `token`: 会话令牌 (必须匹配网关配置)
- `scan_udp_port`: PC期望接收雷达扫描的UDP端口 (0表示不使用)

### Pi 验证

网关检查 token 是否匹配，通过后开始发送 STATUS 帧。

---

## STATUS 帧 (状态上报)

### Pi -> PC (每50ms)

```json
{
  "battery_v": 12.6,
  "control_mode": 0,
  "fault": 0,
  "v_mps": 0.0,
  "omega_radps": 0.0,
  "armed": false,
  "last_sequence": 0,
  "target_v_mps": 0.0,
  "target_omega_radps": 0.0,
  "applied_v_mps": 0.0,
  "applied_omega_radps": 0.0
}
```

- `battery_v`: 电池电压 (V)
- `control_mode`: 控制模式 (0=手动, 1=自动)
- `fault`: 故障码 (0=无故障)
- `v_mps`: 当前线速度 (m/s)
- `omega_radps`: 当前角速度 (rad/s)
- `armed`: 是否已上电授权
- `last_sequence`: 最后处理的命令序号
- `target_*`: 目标速度 (MPPI输出)
- `applied_*`: 实际执行速度 (经过插值)

---

## COMMAND 帧 (控制命令)

### PC -> Pi

```json
{
  "v_mps": 0.15,
  "omega_radps": 0.0,
  "sequence": 1,
  "arm": true
}
```

- `v_mps`: 目标线速度 (m/s)，范围 [-0.05, 0.3]
- `omega_radps`: 目标角速度 (rad/s)，范围 [-0.5, 0.5]
- `sequence`: 命令序号 (单调递增，用于去重)
- `arm`: 是否授权电机 (false=仅仿真，true=实际控制CAN)

---

## 通信流程

```
PC                              Pi 网关
 |                                |
 |------ HELLO {token} --------->|
 |<----- STATUS (每50ms) --------|
 |                                |
 |------ COMMAND {v, ω, seq} -->|
 |<----- STATUS -----------------|
 |                                |
 |------ STOP ------------------>|
 |                                |
```

### 注意事项

1. **超时处理**: 如果5秒内未收到 STATUS 帧，PC 判定连接断开
2. **命令序号**: sequence 必须严格递增，否则被忽略
3. **Watchdog**: Pi 网关有看门狗，超过 0.5s 未收到命令会自动停止

---

## 当前实现

### Pi 端: minimal_gateway4.py

路径: `/tmp/minimal_gateway4.py` (临时文件，需持久化)

功能:
- 监听 TCP 57720 端口
- 处理 HELLO/STATUS/COMMAND 帧
- 记录命令日志 (未集成CAN控制)

TODO:
- [ ] 集成 `ScoutGuardedCanGateway` 发送CAN命令
- [ ] 实现命令插值和平滑
- [ ] 添加状态持久化

### PC 端: simple_pc_controller.py

路径: `D:\mobile-robot-mppi-study-main\simple_pc_controller.py`

功能:
- 连接 Pi 网关 (默认 192.168.43.144:57720)
- 简单的比例控制器 (向目标点移动)
- 发送 COMMAND 帧

用法:
```bash
cd D:\mobile-robot-mppi-study-main
python simple_pc_controller.py --goal-x 1.0 --goal-y 0.0 --duration-s 8
```

参数:
- `--host`: Pi IP (默认 192.168.43.144)
- `--port`: 端口 (默认 57720)
- `--token`: 令牌 (默认 pc_test_token)
- `--goal-x`: 目标X坐标 (默认 2.0)
- `--goal-y`: 目标Y坐标 (默认 0.0)
- `--duration-s`: 运行时长 (默认 10s)
- `--max-v`: 最大线速度 (默认 0.15 m/s)
- `--max-omega`: 最大角速度 (默认 0.2 rad/s)

---

## 故障排查

### 问题: ConnectionRefusedError

```
ConnectionRefusedError: [WinError 10061] 无法建立连接
```

**原因**: Pi 网关未运行或端口不正确

**解决**:
```bash
# 在 Pi 上检查
ps aux | grep minimal_gateway4
ss -tlnp | grep 57720

# 启动网关
python3 /tmp/minimal_gateway4.py
```

### 问题: TimeoutError no status received

```
TimeoutError: no chassis status received from Pi gateway
```

**原因**: 通信协议不匹配 (帧大小错误)

**解决**: 确保两端都使用 `FRAME_HEADER.size` (9字节) 而非硬编码 8

### 问题: CAN Network is down

```
CanOperationError: Failed to transmit: Network is down
```

**原因**: can0 接口未启动

**解决**:
```bash
sudo ip link set can0 up type can bitrate 500000
```

---

## 下一步计划

1. **集成CAN控制到 minimal_gateway4.py**
   - 导入 ScoutGuardedCanGateway
   - 处理 arm 命令
   - 发送运动指令到CAN

2. **替换为完整 MPPI 控制器**
   - 使用 `remote_mppi.py` 作为 PC 端控制器
   - 确保帧格式兼容

3. **持久化网关脚本**
   - 将 minimal_gateway4.py 复制到 `~/rlmppi/`
   - 添加开机自启动

4. **动态避障实车测试**
   - 按照 guide.md 中的场景布置
   - 运行动态障碍物测试

---

## 参考文件

- `guide.md` - 实车测试场景布置指南
- `setup_guide.txt` - 中文版测试布置说明
- `simple_pc_controller.py` - PC端测试控制器
- `/tmp/minimal_gateway4.py` - Pi端简化网关
- `C:\Users\D14\.codex\AGENTS.md` - 环境配置信息
