# ALFWorld 任务

## 目标
在文本模拟的家庭环境中完成指定任务（找东西、清洁、加热等）。

## 输入
- 任务描述: "put a hot potato in the fridge"
- 当前观察: "You are in the middle of a room. Looking around you, you see..."
- 可选动作: ["go to cabinet 1", "open fridge 1", ...]

## 输出
- 选择的动作: "go to cabinet 1"

## 评测指标
- 成功率 = 成功任务数 / 总任务数

## 任务类型（6种）
1. pick_and_place: 拿起物品放到指定位置
2. pick_clean_then_place: 清洁物品后放到指定位置
3. pick_heat_then_place: 加热物品后放到指定位置
4. pick_cool_then_place: 冷却物品后放到指定位置
5. look_at_obj_in_light: 在灯光下查看物品
6. pick_two_obj_and_place: 拿起两个物品放到指定位置
