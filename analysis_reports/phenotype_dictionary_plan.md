# 唑吡坦相关跌倒报告表型词典构建方案

## 1. 构建原则

本研究的跌倒相关表型词典不采用完全主观定义，而采用“权威术语体系 + 药物诱发跌倒文献 + 本地 MedDRA 表核对”的方式构建。

具体原则如下：

1. 术语层级以 MedDRA Preferred Term（PT）为主要分析单位。
2. 如果 FAERS 原始 REAC 中出现 LLT 或历史写法，则依据本地 MedDRA 29.0 表映射到对应 PT。
3. 表型类别按跌倒发生链条拆分为：前驱症状、跌倒事件、跌倒后果。
4. 前驱症状主要覆盖文献中反复提到的药物诱发跌倒相关反应：镇静/嗜睡、意识/认知改变、头晕/眩晕、步态/平衡异常、低血压/体位性低血压、视觉障碍。
5. 跌倒后果主要参考 MedDRA Accidents and injuries (SMQ) 的纳入逻辑，覆盖骨折、损伤、创伤、挫伤、伤口、头部/颅脑损伤、住院和死亡等。

## 2. 依据来源

### 2.1 MedDRA / SMQ 依据

MedDRA 的 Accidents and injuries (SMQ) 明确指出，事故和损伤可与药物使用相关，尤其是老年精神活性药物；该 SMQ 关注由感知、意识、注意力、行为等改变导致的个人事故或损伤。其纳入术语包括 accident、injury、trauma、fall、fracture、wound、crush、contusion 等。

因此，本研究将“跌倒事件”和“跌倒后果”放在同一条报告链条中是有依据的。但需要注意，Accidents and injuries (SMQ) 排除了“事故/损伤的风险因素”类术语，所以头晕、嗜睡、低血压、步态异常等更适合作为“前驱表型”，而不是直接归入事故损伤后果。

### 2.2 药物警戒跌倒研究依据

Rodrigues 等基于葡萄牙药物警戒系统分析药物相关跌倒，使用 MedDRA 术语识别 fall 及可能诱发跌倒的相关 ADR。其表 4 将以下反应作为“可能导致跌倒的 ADR”：hypotension、visual disturbances、gait disorders、dizziness、vertigo、altered state of consciousness、syncope、sleepiness。文中还说明 gait disorders 包括 Gait disturbance、Balance disorder、Coordination abnormal、Mobility decreased、Movement disorder；altered state of consciousness 包括 Altered state of consciousness、Loss of consciousness、Confusional state、Depressed level of consciousness、Disorientation。

Zhou 等基于 FAERS 研究老年药物诱发跌倒，使用 MedDRA PT “fall” 作为纳入条件，并指出老年跌倒与 unsteady gait、balance disturbance、polypharmacy、confusion、visual deficits、cognitive decline 等多因素有关。

### 2.3 唑吡坦药品说明书依据

FDA Ambien/zolpidem 标签中明确提示：唑吡坦可导致 drowsiness 和 decreased level of consciousness，进而导致 falls 和 severe injuries；严重损伤包括 hip fractures 和 intracranial hemorrhage。说明书还报告了 dizziness、drowsiness、falls、confusion 等不良反应，并指出老年患者剂量应降低，以减少 motor/cognitive performance impairment 和对镇静催眠药的敏感性。

### 2.4 老年用药指南依据

2023 AGS Beers Criteria 将 Z-drugs（包括 zolpidem）列为老年人通常应避免药物，理由是其不良事件类似苯二氮卓类，包括 delirium、falls、fractures、急诊/住院增加和机动车事故等。

## 3. 推荐表型词典

### 3.1 前驱症状层

#### A. 镇静/嗜睡/残余镇静类

推荐 PT：

- Somnolence
- Sedation
- Hypersomnia
- Lethargy

可选扩展：

- Fatigue

说明：Fatigue 特异性较弱，建议作为敏感性分析词，不放入主词典。

#### B. 意识/认知改变类

推荐 PT：

- Altered state of consciousness
- Depressed level of consciousness
- Loss of consciousness
- Confusional state
- Disorientation
- Delirium
- Cognitive disorder
- Disturbance in attention
- Memory impairment
- Mental impairment
- Mental status changes

#### C. 头晕/眩晕/晕厥类

推荐 PT：

- Dizziness
- Vertigo
- Vertigo positional
- Vertigo CNS origin
- Vestibular disorder
- Syncope
- Presyncope

可选扩展：

- Vertigo labyrinthine
- Vestibular vertigo

说明：如果本地 FAERS 中这类 PT 出现较少，可以保留在候选词典中，但结果表可合并为“眩晕/前庭相关”。

#### D. 步态/平衡/运动控制异常类

推荐 PT：

- Gait disturbance
- Gait inability
- Balance disorder
- Ataxia
- Coordination abnormal
- Mobility decreased
- Movement disorder

LLT 映射说明：

- Gait abnormal、Gait abnormal NOS、Gait disorder、Gait instability 在本地 MedDRA 表中可映射到 PT Gait disturbance。
- Disequilibrium syndrome 在本地 MedDRA 表中可映射到 PT Balance disorder。

#### E. 低血压/体位性低血压类

推荐 PT：

- Hypotension
- Orthostatic hypotension
- Blood pressure decreased

LLT 映射说明：

- Hypotension orthostatic、Hypotension orthostatic asymptomatic、Hypotension orthostatic symptomatic、Postural hypotension 在本地 MedDRA 表中可映射到 PT Orthostatic hypotension。

#### F. 视觉障碍类

推荐 PT：

- Visual impairment
- Visual acuity reduced
- Vision blurred

LLT 映射说明：

- Visual disturbance、Visual disturbance NOS、Visual disturbances 在本地 MedDRA 表中可映射到 PT Visual impairment。

说明：视觉障碍与跌倒相关文献有依据，但和唑吡坦机制的直接性弱于镇静、认知、眩晕、步态和平衡。建议作为次要前驱表型。

### 3.2 跌倒事件层

推荐主 PT：

- Fall

LLT 映射说明：

- Falling 和 Falling down 在本地 MedDRA 表中均映射到 PT Fall。

说明：论文中可继续保留“Fall/Falling/Falling down”的严格跌倒识别口径，但在 MedDRA PT 结果呈现中应说明其统一归入 PT Fall。

### 3.3 跌倒后果层

#### A. 骨折类

推荐 PT：

- Fracture
- Hip fracture
- Femur fracture

可选扩展：

- Limb fracture
- Spinal fracture
- Vertebral fracture
- Radius fracture
- Wrist fracture

说明：如果希望提高特异性，可只保留 Fracture、Hip fracture、Femur fracture；如果希望提高敏感性，可扩展到所有包含 fracture 的 PT，但需要排除病理性或操作相关骨折。

#### B. 损伤/创伤类

推荐 PT：

- Injury
- Head injury
- Craniocerebral injury
- Contusion
- Wound
- Skin laceration

LLT 映射说明：

- Traumatic brain injury 在本地 MedDRA 表中映射到 PT Craniocerebral injury。
- Laceration 不是单一 PT，建议优先使用 Skin laceration，避免把宫颈裂伤、直肠裂伤、注射部位裂伤等明显不适合跌倒后果的术语纳入。

#### C. 严重结局类

推荐变量：

- Hospitalisation
- Death
- Serious outcome

说明：Hospitalisation 可作为 MedDRA PT/结局字段补充；Death 和 Serious outcome 更建议来自 OUTC/serious 字段，而不是只靠 REAC PT。

## 4. 推荐主词典与敏感性词典

### 4.1 主词典

主词典建议更保守，优先放入文献依据强、MedDRA 映射清楚、和跌倒机制关系直接的 PT：

- Sedation/somnolence phenotype：Somnolence, Sedation, Hypersomnia, Lethargy
- Consciousness/cognition phenotype：Altered state of consciousness, Depressed level of consciousness, Loss of consciousness, Confusional state, Disorientation, Delirium, Cognitive disorder, Disturbance in attention, Memory impairment, Mental impairment, Mental status changes
- Dizziness/vertigo/syncope phenotype：Dizziness, Vertigo, Vertigo positional, Vertigo CNS origin, Vestibular disorder, Syncope, Presyncope
- Gait/balance/motor phenotype：Gait disturbance, Gait inability, Balance disorder, Ataxia, Coordination abnormal, Mobility decreased, Movement disorder
- Hypotension phenotype：Hypotension, Orthostatic hypotension, Blood pressure decreased
- Fall event phenotype：Fall
- Injury/fracture consequence phenotype：Fracture, Hip fracture, Femur fracture, Injury, Head injury, Craniocerebral injury, Contusion, Wound, Skin laceration

### 4.2 敏感性词典

敏感性词典可以增加：

- Fatigue
- Visual impairment
- Visual acuity reduced
- Vision blurred
- Vertigo labyrinthine
- Vestibular vertigo
- 所有包含 fracture 的 PT，但需排除病理性、围手术期或非外伤性骨折

## 5. 可写入论文的方法表述

本研究基于 MedDRA PT 术语体系构建唑吡坦相关跌倒报告表型词典。词典构建参考 MedDRA Accidents and injuries (SMQ)、既往药物警戒跌倒研究、老年药物诱发跌倒相关文献及 zolpidem 药品说明书，并结合本研究所使用的 MedDRA 29.0 中文/英文映射表进行术语核对。根据跌倒发生过程及药理学合理性，表型被分为前驱症状层、跌倒事件层和跌倒后果层。前驱症状层包括镇静/嗜睡、意识/认知改变、头晕/眩晕/晕厥、步态/平衡/运动控制异常、低血压/体位性低血压及视觉障碍；跌倒事件层以 Fall 及其 LLT 变体为核心；跌倒后果层包括骨折、损伤、头部/颅脑损伤、住院及严重结局。对于 FAERS 中出现的 LLT 或历史写法，统一依据 MedDRA 29.0 映射至对应 PT 后进行病例级标记。

## 6. 下一步实现建议

下一步应从原始 REAC 表或可回溯的病例级 REAC 明细中提取所有 PT，而不是只使用现有 fall_pt_list。现有 fall_pt_list 只保存了 broad_fall 词典命中的 PT，不足以识别嗜睡、认知改变、谵妄、镇静等完整前驱表型。

推荐新增一个病例级表型构建脚本，输出以下 0/1 变量：

- pheno_sedation_somnolence
- pheno_consciousness_cognition
- pheno_dizziness_vertigo_syncope
- pheno_gait_balance_motor
- pheno_hypotension
- pheno_visual_disturbance
- pheno_fall_event
- pheno_fracture_injury
- pheno_hospitalisation_or_serious

其中每个变量只要命中该类别下任意一个 PT，即记为 1。
