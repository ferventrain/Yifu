// 通用版全脑分布统计说明文档
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Footer, PageNumber, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, ShadingType, LevelFormat,
} = require("docx");
const fs = require("fs");

const BLACK = "000000";
const GRAY = "595959";
const LINE = 312; // 1.3x

function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: LINE, after: 80 },
    children: [new TextRun({ text, size: 24, color: BLACK })],
    ...opts,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 140, line: LINE },
    children: [new TextRun({ text, bold: true, color: BLACK, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}

function tableCaption(text) {
  return new Paragraph({
    keepNext: true,
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80, line: LINE },
    children: [new TextRun({ text, bold: true, size: 21, color: BLACK, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}

const CELL_MARGINS = { top: 60, bottom: 60, left: 120, right: 120 };
const TABLE_BORDERS = {
  top: { style: BorderStyle.SINGLE, size: 2, color: "9AA6B2" },
  bottom: { style: BorderStyle.SINGLE, size: 2, color: "9AA6B2" },
  left: { style: BorderStyle.NONE },
  right: { style: BorderStyle.NONE },
  insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "D0D0D0" },
  insideVertical: { style: BorderStyle.NONE },
};

function cell(text, { bold = false, width, fill } = {}) {
  return new TableCell({
    children: [new Paragraph({
      spacing: { line: LINE },
      children: [new TextRun({ text, bold, size: 21, color: BLACK })],
    })],
    shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
    margins: CELL_MARGINS,
    width: width ? { size: width, type: WidthType.PERCENTAGE } : undefined,
  });
}

function makeTable(headers, rows, widths) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: TABLE_BORDERS,
    rows: [
      new TableRow({
        tableHeader: true,
        cantSplit: true,
        children: headers.map((t, i) => cell(t, { bold: true, width: widths[i], fill: "F1F5F9" })),
      }),
      ...rows.map(r => new TableRow({
        cantSplit: true,
        children: r.map((t, i) => cell(t, { width: widths[i] })),
      })),
    ],
  });
}

function listItem(text) {
  return new Paragraph({
    numbering: { reference: "list-reading", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: LINE, after: 60 },
    children: [new TextRun({ text, size: 24, color: BLACK })],
  });
}

const doc = new Document({
  creator: "Z.ai",
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: BLACK },
        paragraph: { spacing: { line: LINE } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: BLACK },
        paragraph: { spacing: { before: 320, after: 140, line: LINE } },
      },
    },
  },
  numbering: {
    config: [{
      reference: "list-reading",
      levels: [{
        level: 0,
        format: LevelFormat.DECIMAL,
        text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GRAY })],
        })],
      }),
    },
    children: [
      // 标题区
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 60, line: 460, lineRule: "atLeast" },
        children: [new TextRun({ text: "全脑信号分布统计结果说明", bold: true, size: 36, color: BLACK, font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 160, line: LINE },
        children: [new TextRun({ text: "适用于全脑分布统计系列结果文件（通用版）", size: 21, color: GRAY })],
      }),

      // 一、结果文件概述
      h1("一、结果文件概述"),
      body("本文件为全脑信号定量分布的统计结果。分析时，三维荧光图像数据与标准脑区图谱进行了逐体素配准比对：软件按照图谱的层级解剖结构将全脑划分为不同脑区，再分别统计每个脑区内检出信号的数量、密度与强度，用于回答\u201c信号分布在脑内哪些区域、相对富集程度如何\u201d的问题。"),
      body("结果以 Excel 工作簿交付，可直接使用 Excel 或 WPS 打开查看，无需专用软件。"),

      // 二、工作表结构
      h1("二、工作表结构"),
      body("文件内包含 Level_0 至 Level_9 共十张工作表，对应脑图谱由粗到细的不同解剖层级，见表 1。"),
      tableCaption("表 1　工作表层级结构"),
      makeTable(
        ["工作表", "内容"],
        [
          ["Level_0", "全脑汇总（仅一行），为整个图谱空间内信号的总体统计"],
          ["Level_1", "脑的最大隔室划分，如脑实质（灰质区）、纤维束、脑室系统"],
          ["Level_2 至 Level_4", "逐级细分的大区与亚区，如大脑、小脑、脑干及其下属结构"],
          ["Level_5 至 Level_9", "具体核团与终末分区；层级越深，脑区划分越细"],
        ],
        [25, 75],
      ),
      body("Name 列的格式为\u201c脑区英文名称,标准缩写\u201d，例如 Striatum,STR 表示纹状体，Hypothalamus,HY 表示下丘脑，Corpus callosum,cc 表示胼胝体。缩写可作为快速检索与文献对照的参考。不同文件的具体层数可能因图谱版本与分析设置略有差异，层级命名遵循所用标准脑图谱（如 Allen 脑图谱）的解剖术语。", { spacing: { line: LINE, before: 120, after: 80 } }),

      // 三、数据列说明
      h1("三、数据列说明"),
      body("每张工作表的列结构完全一致，共六列，见表 2。"),
      tableCaption("表 2　数据列说明"),
      makeTable(
        ["列名", "含义", "解读要点"],
        [
          ["Name", "脑区名称及标准缩写", "对应图谱中的解剖分区，用于定位脑区"],
          ["Total Voxels", "该脑区在标准图谱中占据的体素总数", "反映脑区本身体积，是密度计算的分母"],
          ["Signal Voxels", "该脑区内检出信号的体素数", "信号的绝对覆盖范围"],
          ["Voxel Density", "信号体素占比，即 Signal Voxels ÷ Total Voxels", "信号的相对富集密度，已对脑区大小归一化，可直接跨脑区比较"],
          ["Signal Count", "分割得到的信号对象数目（如阳性细胞、信号团块）", "\u201c计数\u201d口径，与密度指标互为补充"],
          ["Sum Intensity", "该脑区内全部信号体素的荧光强度总和", "反映信号总量，数值与脑区大小正相关"],
        ],
        [16, 40, 44],
      ),

      // 四、阅读与使用建议
      h1("四、阅读与使用建议"),
      listItem("跨脑区比较时，建议优先参考 Voxel Density。该指标已消除脑区大小的影响；直接比较 Signal Voxels 或 Sum Intensity 时，体积较大的脑区（如大脑皮层）在数值上天然占优，并不代表其信号密度更高。"),
      listItem("Signal Count 与 Voxel Density 是两种互补口径：前者回答\u201c检出多少个信号对象\u201d，后者回答\u201c信号在该区有多密集\u201d。两者结合可区分\u201c总量大但分布稀疏\u201d与\u201c总量小但高度富集\u201d两类分布模式。"),
      listItem("查找某一特定核团时，建议从中间层级（Level_3 至 Level_5）的工作表入手；需要最细分区的信息时，再查阅更深层级的工作表。"),
      listItem("数值为 0 的行表示该脑区未检出信号，属于正常结果，并非数据缺失。"),
      listItem("体积很小的脑区，其 Voxel Density 对个别信号体素十分敏感，数值波动较大，解读时请结合 Total Voxels 综合判断。"),

      // 五、其他说明
      h1("五、其他说明"),
      body("本说明适用于同系列的全脑分布统计结果文件。不同文件或不同通道（如 ch1、ch2）通常对应不同的荧光标记通道，解读具体数值前，请结合实验方案确认各通道对应的标记物。"),
      body("如需针对特定脑区开展进一步统计、可视化或跨样本比较，可直接基于本文件的分层结构继续分析，也可与我们联系获取技术支持。"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("S:\\Yifu\\全脑信号分布统计结果说明_通用版.docx", buf);
  fs.writeFileSync("S:\\Yifu\\docx_build\\output.docx", buf);
  console.log("DONE");
});
