const W = 1280;
const H = 720;
const TOTAL = 39;

const A = {
  green: "#00512C",
  green2: "#6F947D",
  lightGreen: "#E7F0EA",
  grey: "#EEEEEE",
  text: "#222222",
  muted: "#6B6B6B",
  orange: "#E36C2D",
  blue: "#3466B7",
  red: "#D9362E",
  line: "#CFCFCF",
  logo: "D:/program_FAERS/.codex_scratch/template_ref/media/image2.png",
  campus: "D:/program_FAERS/.codex_scratch/template_ref/media/image1.jpeg",
  annualRor: "D:/program_FAERS/OUTPUT_GLOBAL_COUNTRY/regulatory_trend/figures/annual_ror.png",
  annualRate: "D:/program_FAERS/OUTPUT_GLOBAL_COUNTRY/regulatory_trend/figures/annual_reporting_rate.png",
};

const nav = ["研究背景", "第二章", "第三章", "研究总结"];

const slides = [
  { type: "cover" },
  { type: "declaration" },
  { type: "contents" },
  { type: "section", sec: "01", title: "研 究 背 景" },
  { type: "burden" },
  { type: "clinicalNeed" },
  { type: "evidence" },
  { type: "faersIntro" },
  { type: "mlGap" },
  { type: "gapAim" },
  { type: "section", sec: "02", title: "第二章\n唑吡坦相关跌倒事件的\n药物警戒信号分析" },
  { type: "methodData" },
  { type: "routeSignal" },
  { type: "baseline" },
  { type: "exposureOutcome" },
  { type: "mainResult" },
  { type: "fourMethods" },
  { type: "sensitivity" },
  { type: "trendFigures" },
  { type: "regulatory" },
  { type: "summarySignal" },
  { type: "section", sec: "03", title: "第三章\n老年 FAERS 病例跌倒事件的\n机器学习辅助识别模型" },
  { type: "mlRoute" },
  { type: "features" },
  { type: "splitModels" },
  { type: "mlMetrics" },
  { type: "riskRank" },
  { type: "modelCompare" },
  { type: "importance" },
  { type: "mlExplain" },
  { type: "pvRelation" },
  { type: "mlSummary" },
  { type: "section", sec: "04", title: "研 究 总 结" },
  { type: "conclusions" },
  { type: "innovationLimit" },
  { type: "projectOutputs" },
  { type: "future" },
  { type: "thanks" },
  { type: "end" },
];

function line(fill = "#00000000", width = 0) {
  return { fill, width, style: "solid" };
}

function rect(slide, ctx, x, y, w, h, fill, stroke = "#00000000", name) {
  return ctx.addShape(slide, {
    left: x, top: y, width: w, height: h,
    geometry: "rect",
    fill,
    line: line(stroke, stroke === "#00000000" ? 0 : 1),
    name,
  });
}

function text(slide, ctx, value, x, y, w, h, opt = {}) {
  return ctx.addText(slide, {
    text: String(value ?? ""),
    left: x, top: y, width: w, height: h,
    fontSize: opt.size ?? 18,
    color: opt.color ?? A.text,
    bold: Boolean(opt.bold),
    typeface: opt.face ?? "Microsoft YaHei",
    align: opt.align ?? "left",
    valign: opt.valign ?? "top",
    fill: opt.fill ?? "#00000000",
    line: opt.line ?? line(),
    insets: opt.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    name: opt.name,
  });
}

function rule(slide, ctx, x, y, w, c = A.line, h = 1) {
  rect(slide, ctx, x, y, w, h, c);
}

async function header(slide, ctx, page, active = 0) {
  rect(slide, ctx, 0, 0, W, 58, A.grey);
  await ctx.addImage(slide, { path: A.logo, left: 18, top: 4, width: 194, height: 50, fit: "contain", alt: "Sun Yat-sen University" });
  const xs = [270, 470, 700, 930];
  nav.forEach((n, i) => {
    if (i === active) {
      rect(slide, ctx, xs[i], 0, 190, 58, A.green);
      text(slide, ctx, n, xs[i], 12, 190, 34, { size: 28, color: "#FFFFFF", bold: true, align: "center" });
    } else {
      text(slide, ctx, n, xs[i], 16, 190, 28, { size: 24, color: "#888888", bold: true, align: "center" });
    }
  });
  rect(slide, ctx, 0, 57, W, 1, "#C8C8C8");
  // subtle shadow band like the reference deck
  rect(slide, ctx, 0, 58, W, 8, "#D5D5D5");
  footer(slide, ctx, page);
}

function footer(slide, ctx, page, source = "资料来源：FAERS 2004-2025 数据处理结果、项目论文初稿及模型输出。") {
  text(slide, ctx, source, 68, 695, 780, 18, { size: 9, color: A.muted });
  text(slide, ctx, `${page}/${TOTAL}`, 1155, 688, 70, 22, { size: 14, color: A.muted, align: "right" });
}

function pageTitle(slide, ctx, title, x = 36, y = 82, w = 620) {
  rect(slide, ctx, x - 18, y, w, 64, A.green);
  const compact = title.length > 20;
  text(slide, ctx, title, x, y + 7, w - 34, 50, { size: compact ? 21 : 26, color: "#FFFFFF", bold: true, valign: "middle" });
}

function note(slide, ctx, value, x, y, w, h, color = A.green) {
  rect(slide, ctx, x, y, w, h, "#FFFFFF", color);
  text(slide, ctx, value, x + 14, y + 12, w - 28, h - 24, { size: 17, bold: true });
}

function metric(slide, ctx, value, label, x, y, color = A.green) {
  text(slide, ctx, value, x - 52, y, 264, 46, { size: 34, color, bold: true, face: "Georgia", align: "center" });
  text(slide, ctx, label, x - 50, y + 48, 260, 38, { size: 14, bold: true, align: "center" });
}

function bullet(slide, ctx, items, x, y, w, size = 21, gap = 38) {
  items.forEach((it, i) => {
    text(slide, ctx, "➢", x, y + i * gap, 28, 24, { size, color: A.green, bold: true });
    text(slide, ctx, it, x + 30, y + i * gap, w, 32, { size, bold: true });
  });
}

function miniTable(slide, ctx, x, y, cols, rows, widths, rowH = 36) {
  rect(slide, ctx, x, y, widths.reduce((a, b) => a + b, 0), rowH, A.green);
  let xx = x;
  cols.forEach((c, i) => {
    text(slide, ctx, c, xx + 6, y + 8, widths[i] - 12, 20, { size: 13, color: "#FFFFFF", bold: true, align: "center" });
    xx += widths[i];
  });
  rows.forEach((r, ri) => {
    xx = x;
    const yy = y + rowH * (ri + 1);
    rect(slide, ctx, x, yy, widths.reduce((a, b) => a + b, 0), rowH, ri % 2 ? "#F5F8F6" : "#FFFFFF", "#DDDDDD");
    r.forEach((c, i) => {
      text(slide, ctx, c, xx + 6, yy + 8, widths[i] - 12, 20, { size: 12.5, align: "center" });
      xx += widths[i];
    });
  });
}

function flowBox(slide, ctx, label, x, y, w, h, stroke = A.green, fill = "#FFFFFF") {
  rect(slide, ctx, x, y, w, h, fill, stroke);
  text(slide, ctx, label, x + 8, y + 8, w - 16, h - 16, { size: 15, bold: true, align: "center", valign: "middle" });
}

function arrow(slide, ctx, x1, y1, x2, y2, color = "#333333") {
  if (Math.abs(x2 - x1) >= Math.abs(y2 - y1)) {
    const x = Math.min(x1, x2);
    rule(slide, ctx, x, y1, Math.abs(x2 - x1), color, 2);
  } else {
    const y = Math.min(y1, y2);
    rect(slide, ctx, x1, y, 2, Math.abs(y2 - y1), color);
  }
  rect(slide, ctx, x2 - 4, y2 - 4, 8, 8, color);
}

async function section(slide, ctx, spec) {
  rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
  rect(slide, ctx, 0, 0, 170, 170, "#E3ECE7");
  rect(slide, ctx, 0, 0, 110, 110, "#6F947D");
  rect(slide, ctx, 0, 690, 580, 1, "#999999");
  rect(slide, ctx, 1010, 575, 170, 135, "#E3ECE7");
  rect(slide, ctx, 1060, 625, 95, 70, A.green);
  await ctx.addImage(slide, { path: A.logo, left: 535, top: 658, width: 210, height: 48, fit: "contain", alt: "logo" });
  text(slide, ctx, spec.sec, 420, 260, 80, 60, { size: 36, color: A.green, bold: true, face: "Georgia", align: "center" });
  text(slide, ctx, spec.title, 505, 250, 520, 120, { size: 34, bold: true, color: A.text, valign: "middle" });
}

async function drawCover(slide, ctx) {
  rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
  await ctx.addImage(slide, { path: A.logo, left: 480, top: 36, width: 320, height: 86, fit: "contain", alt: "logo" });
  await ctx.addImage(slide, { path: A.campus, left: 0, top: 172, width: W, height: 354, fit: "cover", alt: "campus" });
  rect(slide, ctx, 0, 172, W, 354, "#00512CDD");
  text(slide, ctx, "基于 FAERS 数据库的老年人唑吡坦相关跌倒事件药物警戒研究", 96, 260, 1080, 154, {
    size: 48, color: "#FFFFFF", bold: true,
  });
  const meta = ["汇报人：", "导   师：", "专   业：公共卫生", "日   期：2026年5月25日"];
  meta.forEach((m, i) => text(slide, ctx, m, 540, 558 + i * 28, 300, 26, { size: 23, color: A.green, bold: true }));
}

function contents(slide, ctx) {
  rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
  rect(slide, ctx, 0, 0, 150, 150, A.green2);
  text(slide, ctx, "目录", 972, 72, 220, 90, { size: 70, color: A.green, bold: true });
  const items = [
    ["01", "研 究 背 景"],
    ["02", "唑吡坦相关跌倒事件的\n药物警戒信号分析"],
    ["03", "老年 FAERS 病例跌倒事件的\n机器学习辅助识别模型"],
    ["04", "研 究 总 结"],
    ["05", "致 谢"],
  ];
  items.forEach((it, i) => {
    const y = 110 + i * 114;
    rect(slide, ctx, 342, y, 68, 68, A.green2);
    text(slide, ctx, it[0], 342, y + 14, 68, 34, { size: 26, color: "#FFFFFF", bold: true, face: "Georgia", align: "center" });
    text(slide, ctx, it[1], 424, y + 10, 500, 55, { size: i === 0 || i > 2 ? 30 : 27, bold: true });
  });
  rect(slide, ctx, 0, 690, 580, 1, "#999999");
  rect(slide, ctx, 1005, 580, 185, 130, "#E3ECE7");
  rect(slide, ctx, 1085, 635, 95, 70, A.green);
  text(slide, ctx, "SUN YAT-SEN UNIVERSITY", 535, 680, 220, 18, { size: 10, color: A.green, align: "center" });
}

async function normal(slide, ctx, page, active, titleText, contentFn) {
  rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
  await header(slide, ctx, page, active);
  if (titleText) pageTitle(slide, ctx, titleText);
  await contentFn();
}

export async function buildSlide(presentation, ctx, page) {
  const spec = slides[page - 1];
  const slide = presentation.slides.add();
  if (spec.type === "cover") {
    await drawCover(slide, ctx);
    return slide;
  }
  if (spec.type === "declaration") {
    rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
    footer(slide, ctx, page, "");
    text(slide, ctx, "学位论文原创性声明", 360, 94, 560, 50, { size: 34, color: A.green, bold: true, align: "center" });
    text(slide, ctx, "本人郑重声明：所呈交的学位论文，是本人在导师指导下，独立进行研究工作所取得的成果。除文中已经注明引用的内容外，本论文不包含其他个人或集体已经发表或撰写过的作品成果。对本文研究作出重要贡献的个人和集体，均已在文中以明确方式标明。本人完全意识到本声明的法律结果由本人承担。", 170, 220, 940, 260, { size: 24, color: "#333333", valign: "middle" });
    return slide;
  }
  if (spec.type === "contents") {
    contents(slide, ctx);
    return slide;
  }
  if (spec.type === "section") {
    await section(slide, ctx, spec);
    return slide;
  }
  if (spec.type === "thanks") {
    rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
    footer(slide, ctx, page, "");
    text(slide, ctx, "致        谢", 470, 100, 340, 56, { size: 38, color: A.green, bold: true, align: "center" });
    bullet(slide, ctx, ["感谢导师在课题设计、数据分析和论文写作中的指导。", "感谢课题组老师和同学在数据整理、代码实现和结果讨论中的帮助。", "感谢各位专家老师莅临指导，恳请批评指正。"], 210, 255, 800, 23, 70);
    return slide;
  }
  if (spec.type === "end") {
    rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
    await ctx.addImage(slide, { path: A.logo, left: 462, top: 54, width: 356, height: 90, fit: "contain", alt: "logo" });
    text(slide, ctx, "恳请各位老师批评指正！", 320, 300, 640, 70, { size: 42, color: A.green, bold: true, align: "center" });
    text(slide, ctx, "答辩人：        指导老师：", 430, 470, 420, 30, { size: 24, color: A.green, bold: true, align: "center" });
    return slide;
  }
  const active = page < 11 ? 0 : page < 22 ? 1 : page < 34 ? 2 : 3;
  const titleMap = {
    burden: "老年跌倒与用药安全",
    clinicalNeed: "唑吡坦用药后的跌倒风险关注",
    evidence: "既往指南和监管信息提示",
    faersIntro: "FAERS 数据库与药物警戒信号",
    mlGap: "机器学习的辅助定位",
    gapAim: "研究不足与优化目标",
    methodData: "第二章 研究方法",
    routeSignal: "第二章 技术路线图",
    baseline: "全周期数据集构建与总体样本",
    exposureOutcome: "暴露与结局分布",
    mainResult: "主分析结果",
    fourMethods: "四种方法均提示阳性信号",
    sensitivity: "敏感性分析与补充分析方向一致",
    trendFigures: "年度趋势分析",
    regulatory: "监管节点前后，主信号方向没有反转",
    summarySignal: "第二章 讨论小结",
    mlRoute: "第三章 研究方法",
    features: "特征工程",
    splitModels: "模型训练与验证策略",
    mlMetrics: "XGBoost 测试集表现",
    riskRank: "高风险分层识别能力",
    modelCompare: "模型定位与解释边界",
    importance: "重要特征与药物警戒解释",
    mlExplain: "模型解释原则",
    pvRelation: "机器学习与传统信号检测",
    mlSummary: "第三章 讨论小结",
    conclusions: "主要结论",
    innovationLimit: "创新性与局限性",
    projectOutputs: "项目实现与阶段性成果",
    future: "后续研究方向",
    thanks: "致        谢",
    end: "",
  };

  await normal(slide, ctx, page, active, titleMap[spec.type], async () => {
    if (spec.type === "burden") {
      note(slide, ctx, "老年人跌倒可导致骨折、住院、功能下降甚至死亡，是可预防伤害中的重点问题。", 70, 160, 520, 110, A.green);
      metric(slide, ctx, "65+", "研究对象：老年病例", 715, 160);
      metric(slide, ctx, "4,155,023", "FAERS 全周期病例", 1010, 160, A.orange);
      bullet(slide, ctx, ["失眠治疗不能只看“能否入睡”，还要看第二天活动能力和跌倒风险。", "多重用药、合并慢病和中枢神经系统药物使用，会让老年患者风险更复杂。", "药物相关跌倒属于临床上相对可干预的风险点。"], 90, 330, 900);
    }
    if (spec.type === "clinicalNeed") {
      flowBox(slide, ctx, "失眠", 90, 210, 150, 58); arrow(slide, ctx, 240, 238, 330, 238, A.green);
      flowBox(slide, ctx, "唑吡坦\nZ-drug", 330, 200, 170, 78, A.orange); arrow(slide, ctx, 500, 238, 600, 238, A.green);
      flowBox(slide, ctx, "镇静、头晕\n反应迟钝", 600, 200, 190, 78); arrow(slide, ctx, 790, 238, 890, 238, A.green);
      flowBox(slide, ctx, "跌倒/骨折\n风险关注", 890, 200, 190, 78, A.red);
      bullet(slide, ctx, ["FDA 曾因次日损害风险要求降低部分制剂推荐剂量。", "2019 年 Z-drugs 被要求加入复杂睡眠行为相关严重伤害风险黑框警告。", "因此需要从上市后真实报告中观察老年人跌倒相关信号。"], 110, 380, 940);
    }
    if (spec.type === "evidence") {
      const cards = [["指南共识", "Beers Criteria 和失眠指南均强调老年镇静催眠药使用需谨慎。"], ["既往研究", "Z-drugs 与跌倒、骨折、损伤风险的关联在多项研究中被提示。"], ["证据空白", "针对 FAERS 老年病例、全周期和机器学习辅助排序的系统整理仍有限。"]];
      cards.forEach((c, i) => { rect(slide, ctx, 90 + i * 360, 210, 300, 220, i === 1 ? A.green : "#F4F7F5", i === 1 ? A.green : A.green2); text(slide, ctx, c[0], 120 + i * 360, 242, 240, 36, { size: 28, color: i === 1 ? "#FFFFFF" : A.green, bold: true }); text(slide, ctx, c[1], 120 + i * 360, 315, 240, 70, { size: 18, color: i === 1 ? "#FFFFFF" : A.text, bold: true }); });
      text(slide, ctx, "本研究定位：不是直接证明因果，而是识别和刻画药物警戒报告信号。", 110, 525, 890, 34, { size: 24, bold: true });
    }
    if (spec.type === "faersIntro") {
      miniTable(slide, ctx, 110, 180, ["数据表", "本研究用途"], [["DEMO", "病例去重、年龄、性别、报告年份"], ["DRUG", "识别唑吡坦、其他 Z-drugs、合并用药"], ["REAC", "构建狭义/广义跌倒相关结局"], ["OUTC", "提取住院、死亡等严重结局"]], [180, 720], 48);
      text(slide, ctx, "FAERS 的优势是覆盖面大、适合早期发现信号；短板是不能计算真实发生率，也不能单独证明因果。", 130, 555, 930, 36, { size: 22, bold: true });
    }
    if (spec.type === "mlGap") {
      note(slide, ctx, "传统信号检测回答“是否有异常报告优势”；机器学习回答“哪些病例更值得优先关注”。", 80, 185, 500, 120, A.green);
      note(slide, ctx, "本研究把机器学习定位为辅助排序工具，不把模型结果解释为临床因果预测。", 700, 185, 420, 120, A.orange);
      miniTable(slide, ctx, 170, 380, ["任务", "对应问题", "输出"], [["信号检测", "唑吡坦-跌倒是否存在报告优势", "ROR/PRR/IC/EBGM"], ["机器学习", "病例层面谁更像跌倒相关报告", "风险排序、特征重要性"]], [160, 430, 280], 48);
    }
    if (spec.type === "gapAim") {
      miniTable(slide, ctx, 100, 170, ["研究不足", "本研究优化目标"], [["既往研究多关注临床队列或单一方法", "构建 2004-2025 年 FAERS 全周期病例级数据集"], ["Z-drugs 老年跌倒报告信号需要多口径验证", "联合 ROR、PRR、IC、EBGM 和敏感性分析"], ["机器学习在药物警戒中解释边界易被放大", "明确其只用于辅助筛查和风险排序"]], [420, 600], 78);
    }
    if (spec.type === "methodData") {
      miniTable(slide, ctx, 70, 150, ["项目", "定义"], [["研究对象", "2004-2025 年 FAERS 中 65 岁及以上老年病例"], ["主暴露", "唑吡坦被报告为 primary suspect 或 secondary suspect"], ["主结局", "狭义跌倒事件：FALL、FALLING 等直接跌倒术语"], ["补充结局", "广义跌倒相关事件：眩晕、平衡障碍、低血压等"], ["协变量", "年龄、性别、年份、季度、多重用药及 CNS 合并用药"]], [230, 850], 48);
    }
    if (spec.type === "routeSignal") {
      flowBox(slide, ctx, "FAERS 季度原始数据", 500, 150, 240, 42); flowBox(slide, ctx, "DEMO 去重", 95, 245, 190, 46); flowBox(slide, ctx, "DRUG 暴露", 365, 245, 190, 46); flowBox(slide, ctx, "REAC 结局", 635, 245, 190, 46); flowBox(slide, ctx, "OUTC 严重结局", 905, 245, 190, 46);
      rule(slide, ctx, 190, 220, 810, A.green2, 3); [190, 460, 730, 1000].forEach(x => rect(slide, ctx, x, 220, 3, 48, A.green2)); rect(slide, ctx, 620, 192, 3, 28, A.green2);
      flowBox(slide, ctx, "病例级主分析表", 500, 355, 240, 52, A.green, A.lightGreen); rect(slide, ctx, 620, 291, 3, 64, A.green2);
      ["不成比例分析", "敏感性分析", "年度趋势", "校正模型"].forEach((s, i) => flowBox(slide, ctx, s, 190 + i * 230, 430, 170, 58, i === 0 ? A.orange : A.green2));
    }
    if (spec.type === "baseline") {
      metric(slide, ctx, "4,155,023", "老年病例", 110, 185, A.green);
      metric(slide, ctx, "132,266", "狭义跌倒病例", 390, 185, A.orange);
      metric(slide, ctx, "298,906", "广义跌倒相关病例", 660, 185, A.blue);
      metric(slide, ctx, "7,798", "唑吡坦 only PS/SS 暴露", 940, 185, A.red);
      bullet(slide, ctx, ["全周期去重后形成病例级索引，避免同一病例重复进入分析。", "主分析排除唑吡坦与其他 Z-drugs 混合暴露，减少解释困难。", "样本量支持总体信号、年度趋势和多个补充分析。"], 110, 420, 920);
    }
    if (spec.type === "exposureOutcome") {
      miniTable(slide, ctx, 120, 160, ["指标", "数量"], [["唑吡坦 only PS/SS 暴露病例", "7,798"], ["其他 Z-drug only PS/SS 暴露病例", "6,799"], ["唑吡坦 suspect 暴露病例", "8,011"], ["狭义跌倒事件", "132,266"], ["广义跌倒相关事件", "298,906"]], [540, 280], 56);
      text(slide, ctx, "这页用于说明：主分析人群和结局不是临时挑出来的，而是在清洗流程中预先定义。", 145, 575, 880, 30, { size: 22, bold: true });
    }
    if (spec.type === "mainResult") {
      metric(slide, ctx, "13.38%", "唑吡坦暴露组狭义跌倒报告比例", 120, 190, A.red);
      metric(slide, ctx, "3.16%", "参照组狭义跌倒报告比例", 420, 190, A.green);
      metric(slide, ctx, "ROR 4.73", "95%CI 4.43-5.05", 720, 190, A.orange);
      metric(slide, ctx, "PRR 4.23", "95%CI 3.99-4.47", 990, 190, A.blue);
      text(slide, ctx, "大白话：在唑吡坦相关报告中，跌倒被报告出来的比例明显更高。", 140, 470, 850, 38, { size: 27, bold: true, color: A.green });
    }
    if (spec.type === "fourMethods") {
      miniTable(slide, ctx, 140, 170, ["方法", "主要结果", "信号判断"], [["ROR", "4.73（95%CI 4.43-5.05）", "阳性"], ["PRR", "4.23（95%CI 3.99-4.47）", "阳性"], ["IC", "2.07，IC025 > 0", "阳性"], ["EBGM", "4.19，EB05 ≥ 2", "阳性"]], [180, 520, 180], 62);
      text(slide, ctx, "四种方法都提示阳性，说明结论不是某一种算法“带偏”的结果。", 170, 585, 780, 30, { size: 23, bold: true });
    }
    if (spec.type === "sensitivity") {
      miniTable(slide, ctx, 100, 170, ["分析口径", "结果方向", "解释"], [["广义跌倒相关事件", "阳性", "扩大结局定义后仍同向"], ["PS only 暴露定义", "阳性", "更严格暴露定义下仍提示信号"], ["与其他 Z-drugs 比较", "补充支持", "用于观察唑吡坦与同类药物差异"], ["分层探索", "部分背景风险更高", "高龄、多重用药、CNS 合并用药更值得关注"]], [260, 180, 560], 62);
    }
    if (spec.type === "trendFigures") {
      await ctx.addImage(slide, { path: A.annualRor, left: 72, top: 145, width: 530, height: 330, fit: "contain", alt: "annual ROR" });
      await ctx.addImage(slide, { path: A.annualRate, left: 650, top: 145, width: 530, height: 330, fit: "contain", alt: "annual reporting rate" });
      text(slide, ctx, "按年拆开后，多数年份仍保持阳性信号；年份间有波动，但总体方向没有消失。", 100, 535, 900, 34, { size: 23, bold: true });
    }
    if (spec.type === "regulatory") {
      miniTable(slide, ctx, 120, 165, ["监管节点", "时期", "ROR", "跌倒报告比例"], [["2013 FDA 剂量调整", "调整前", "4.84", "15.46%"], ["2013 FDA 剂量调整", "调整后", "4.63", "12.83%"], ["2019 FDA 黑框警告", "警告前", "4.44", "13.41%"], ["2019 FDA 黑框警告", "警告后", "5.02", "13.32%"]], [300, 160, 160, 220], 60);
      text(slide, ctx, "监管节点前后均为同向阳性，比较像长期存在的药物警戒现象。", 150, 565, 850, 30, { size: 23, bold: true });
    }
    if (spec.type === "summarySignal") {
      bullet(slide, ctx, ["唑吡坦 suspect 暴露与狭义跌倒报告之间存在明显不成比例报告信号。", "信号在广义结局、PS only 暴露和年度趋势中方向一致。", "结果应解释为报告信号，不能单独证明因果关系。"], 100, 175, 960, 25, 60);
      note(slide, ctx, "临床启示：老年患者使用唑吡坦时，应特别关注高龄、多重用药和合并中枢神经系统药物使用。", 150, 470, 850, 90, A.green);
    }
    if (spec.type === "mlRoute") {
      flowBox(slide, ctx, "病例级特征表\n2004-2025", 80, 185, 190, 78); arrow(slide, ctx, 270, 224, 370, 224, A.green);
      flowBox(slide, ctx, "训练集\n≤2023", 370, 165, 150, 58); flowBox(slide, ctx, "验证集\n2024", 370, 255, 150, 58); arrow(slide, ctx, 520, 224, 640, 224, A.green);
      flowBox(slide, ctx, "模型训练\nLR/RF/XGBoost", 640, 185, 210, 78, A.orange); arrow(slide, ctx, 850, 224, 950, 224, A.green);
      flowBox(slide, ctx, "测试集\n2025", 950, 185, 150, 78);
      bullet(slide, ctx, ["目标变量：狭义跌倒事件。", "评价指标：ROC-AUC、Average precision、高风险分层检出比例。", "定位：辅助识别和风险排序，而不是临床因果预测。"], 120, 420, 900);
    }
    if (spec.type === "features") {
      miniTable(slide, ctx, 90, 155, ["特征类型", "示例"], [["人口学与时间", "年龄组、性别、报告年份、季度、国家"], ["药物暴露", "唑吡坦、其他 Z-drugs、药物角色、用药负担"], ["合并用药", "苯二氮䓬类、抗抑郁药、抗精神病药、阿片类、抗癫痫药"], ["适应证/报告来源", "失眠、疼痛、焦虑、报告者类型等"], ["MedDRA 相关特征", "系统器官分类和术语映射"]], [260, 780], 52);
    }
    if (spec.type === "splitModels") {
      miniTable(slide, ctx, 130, 175, ["模型", "用途", "特点"], [["Logistic regression", "基线模型", "可解释性强"], ["Random forest", "非线性模型", "可捕捉变量交互"], ["XGBoost", "主力模型", "在测试集中表现最好"]], [260, 380, 260], 64);
      text(slide, ctx, "按年份外推验证：≤2023 训练、2024 验证、2025 测试，更接近真实药物警戒监测场景。", 150, 540, 930, 34, { size: 22, bold: true });
    }
    if (spec.type === "mlMetrics") {
      metric(slide, ctx, "0.772", "ROC-AUC", 210, 210, A.green);
      metric(slide, ctx, "0.119", "Average precision", 510, 210, A.orange);
      metric(slide, ctx, "14.79%", "前 5% 高风险病例跌倒比例", 820, 210, A.red);
      text(slide, ctx, "模型有一定区分能力，尤其适合把病例按风险排序后优先查看。", 170, 470, 880, 36, { size: 26, bold: true });
    }
    if (spec.type === "riskRank") {
      miniTable(slide, ctx, 150, 170, ["分层", "狭义跌倒事件比例", "相对总体阳性比例"], [["测试集总体", "约 3.15%", "1.00 倍"], ["前 10% 高风险病例", "升高", "高于总体"], ["前 5% 高风险病例", "14.79%", "约 4.70 倍"]], [260, 300, 300], 70);
      text(slide, ctx, "大白话：模型不能告诉我们“谁一定会跌倒”，但能帮我们把更值得关注的报告排到前面。", 150, 555, 900, 32, { size: 22, bold: true });
    }
    if (spec.type === "modelCompare") {
      note(slide, ctx, "传统药物警戒：回答药物-事件组合是否异常。", 120, 170, 410, 120, A.green);
      note(slide, ctx, "机器学习：回答病例级别哪些报告更像跌倒相关。", 710, 170, 410, 120, A.orange);
      miniTable(slide, ctx, 170, 380, ["不能做的事", "原因"], [["不能替代 ROR/PRR/IC/EBGM", "模型不是信号判定标准"], ["不能直接证明因果", "FAERS 缺少真实用药分母和完整临床过程"], ["不能直接外推到临床预测", "需要医保/EHR/前瞻性数据验证"]], [360, 520], 54);
    }
    if (spec.type === "importance") {
      miniTable(slide, ctx, 110, 160, ["特征方向", "可能解释"], [["高龄", "基础跌倒风险升高，药物镇静效应更容易放大"], ["多重用药", "药物相互作用和用药负担增加"], ["抗抑郁药/阿片类/抗癫痫药", "均可能影响中枢神经系统和平衡功能"], ["唑吡坦 suspect 暴露", "与传统信号检测结果方向一致"]], [330, 650], 64);
    }
    if (spec.type === "mlExplain") {
      bullet(slide, ctx, ["模型解释要看是否符合临床药理和老年跌倒机制。", "变量重要性不是因果效应大小，不能和回归 OR 直接等同。", "当模型结果和传统信号检测一致时，可作为补充证据；不一致时要回到数据质量和定义口径。"], 110, 180, 930, 24, 62);
    }
    if (spec.type === "pvRelation") {
      flowBox(slide, ctx, "信号检测\n群体层面", 130, 220, 190, 90, A.green); arrow(slide, ctx, 320, 265, 515, 265, A.green);
      flowBox(slide, ctx, "机器学习\n病例层面", 515, 220, 190, 90, A.orange); arrow(slide, ctx, 705, 265, 900, 265, A.green);
      flowBox(slide, ctx, "后续验证\n真实世界数据", 900, 220, 190, 90, A.blue);
      text(slide, ctx, "三者关系：发现信号 → 排序病例 → 外部验证。每一步回答的问题不一样。", 140, 450, 880, 38, { size: 27, bold: true, color: A.green });
    }
    if (spec.type === "mlSummary") {
      bullet(slide, ctx, ["XGBoost 在 2025 测试集中 ROC-AUC 为 0.772，具有一定排序能力。", "前 5% 高风险病例狭义跌倒比例达到 14.79%，约为总体阳性比例的 4.70 倍。", "机器学习适合作为药物警戒辅助筛查工具，但不能替代传统信号检测。"], 105, 180, 960, 25, 62);
    }
    if (spec.type === "conclusions") {
      const rows = [["01", "FAERS 老年病例中，唑吡坦相关报告存在清楚的跌倒相关不成比例报告信号。"], ["02", "信号在多种统计方法、结局定义、暴露定义和年度趋势中方向一致。"], ["03", "高龄、多重用药和合并中枢神经系统药物使用是解释和临床关注的重点背景。"], ["04", "机器学习可用于病例优先排序，但不能直接解释为临床因果预测。"]];
      rows.forEach((r, i) => { text(slide, ctx, r[0], 105, 150 + i * 105, 60, 40, { size: 30, color: i % 2 ? A.orange : A.green, bold: true, face: "Georgia" }); text(slide, ctx, r[1], 180, 150 + i * 105, 880, 54, { size: 22, bold: true }); rule(slide, ctx, 180, 210 + i * 105, 860, A.line); });
    }
    if (spec.type === "innovationLimit") {
      miniTable(slide, ctx, 80, 145, ["创新性", "局限性"], [["构建 2004-2025 年 FAERS 老年病例全周期数据集", "自发报告系统存在漏报、重复报告和报告偏倚"], ["同时采用多种不成比例分析和敏感性分析", "缺少处方总人数，不能计算真实发生率"], ["引入机器学习辅助病例排序并明确解释边界", "仍需医保、EHR 或前瞻性研究进一步验证"]], [560, 560], 82);
    }
    if (spec.type === "projectOutputs") {
      bullet(slide, ctx, ["完成 FAERS 季度数据清洗、病例级数据集构建和全周期去重。", "形成 signal_dataset、drug_feature、global_case_index 等可复用 Parquet 数据集。", "输出年度趋势图、监管节点比较表、国家分布报告和机器学习结果。", "撰写论文初稿，并整理参考文献与方法学说明。"], 110, 170, 980, 24, 60);
    }
    if (spec.type === "future") {
      note(slide, ctx, "真实世界验证", 110, 170, 280, 90, A.green); note(slide, ctx, "更细药物剂量和时序", 500, 170, 280, 90, A.orange); note(slide, ctx, "模型外部验证与校准", 890, 170, 280, 90, A.blue);
      bullet(slide, ctx, ["使用医保数据库或电子病历验证跌倒/骨折风险。", "进一步区分剂型、剂量、用药时间和联合用药时序。", "将机器学习模型作为筛查工具进行外部验证，而不是直接临床应用。"], 130, 360, 900);
    }
    if (spec.type === "thanks") {
      rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
      footer(slide, ctx, page, "");
      text(slide, ctx, "致        谢", 470, 100, 340, 56, { size: 38, color: A.green, bold: true, align: "center" });
      bullet(slide, ctx, ["感谢导师在课题设计、数据分析和论文写作中的指导。", "感谢课题组老师和同学在数据整理、代码实现和结果讨论中的帮助。", "感谢各位专家老师于百忙之中莅临指导，恳请批评指正。"], 210, 255, 800, 23, 70);
    }
    if (spec.type === "end") {
      rect(slide, ctx, 0, 0, W, H, "#FFFFFF");
      await ctx.addImage(slide, { path: A.logo, left: 462, top: 54, width: 356, height: 90, fit: "contain", alt: "logo" });
      text(slide, ctx, "恳请各位老师批评指正！", 320, 300, 640, 70, { size: 42, color: A.green, bold: true, align: "center" });
      text(slide, ctx, "答辩人：        指导老师：", 430, 470, 420, 30, { size: 24, color: A.green, bold: true, align: "center" });
    }
  });
  return slide;
}
