/* 电商会话推荐 · 闭环实验可视化 */
(function () {
  'use strict';

  const DATA = window.__DATA__;
  const LABELS = { SRGNN: 'SR-GNN', GCSAN: 'GC-SAN', GCEGNN: 'GCE-GNN' };
  const C = {
    ink: '#16272B', ink2: '#3D4C4E', muted: '#7B7669', line: '#D7CEBD',
    paper: '#F7F2E8', paper2: '#EFE7D8', vermillion: '#C24B2F', teal: '#1D6F76',
    gold: '#A97F2A', green: '#2E7D4F', red: '#B23A2E', gray: '#8A8F98'
  };
  const MODEL_COLOR = { SRGNN: C.gray, GCSAN: C.teal, GCEGNN: C.vermillion };
  const charts = [];
  const chartInstances = {};
  const state = { caseIndex: 0, model: 'GCEGNN', graphMode: 'local' };

  const $ = (sel) => document.querySelector(sel);
  const fmt = (n) => Number(n).toLocaleString('en-US');
  const pct = (x, d = 2) => (Number(x) * 100).toFixed(d) + '%';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  function baseOption() {
    return {
      textStyle: { fontFamily: '"Microsoft YaHei", "PingFang SC", sans-serif', color: C.ink },
      tooltip: {
        backgroundColor: C.ink, borderWidth: 0, padding: [9, 12],
        textStyle: { color: C.paper, fontSize: 12 },
        extraCssText: 'box-shadow:0 8px 24px rgba(22,39,43,.18);border-radius:2px;'
      },
      grid: { left: 58, right: 30, top: 42, bottom: 48, containLabel: true },
      animationDuration: 700,
      animationEasing: 'cubicOut'
    };
  }

  function makeChart(el, key) {
    if (chartInstances[key]) return chartInstances[key];
    const chart = echarts.init(el, null, { renderer: 'canvas' });
    chartInstances[key] = chart;
    charts.push(chart);
    return chart;
  }

  function emptyState(el, text) {
    el.innerHTML = '<div class="chart-empty">' + text + '</div>';
  }

  /* ───────────── 封面 ───────────── */
  function initCover() {
    const m = DATA.meta;
    const cells = [
      [fmt(m.counts.sessions), 'TEST SESSIONS'],
      [fmt(m.counts.items), 'CANDIDATE ITEMS'],
      [fmt(m.funnel.selected_interactions), 'INTERACTIONS'],
      [String(DATA.metrics.length), 'MODELS']
    ];
    $('#specimenNumbers').innerHTML = cells.map(([v, l]) =>
      `<div><b>${v}</b><small>${l}</small></div>`).join('');

    const first = DATA.cases.find(c => c.kind === 'success') || DATA.cases[0];
    if (!first) return;
    const h = first.history;
    let parts;
    if (h.length <= 6) {
      parts = h.map(x => `<span class="chip">${x}</span>`);
    } else {
      parts = [
        `<span class="chip">${h[0]}</span>`,
        `<span class="chip">${h[1]}</span>`,
        `<span class="arrow">···</span>`,
        `<span class="chip">${h[h.length - 2]}</span>`,
        `<span class="chip">${h[h.length - 1]}</span>`
      ];
    }
    parts.push('<span class="arrow">→</span>');
    parts.push(`<span class="chip target">${first.target}</span>`);
    $('#heroStrip').innerHTML = parts.join('');
  }

  /* ───────────── 01 全景 ───────────── */
  function initOverview() {
    const f = DATA.meta.funnel;
    const max = f.source_sessions;
    const steps = [
      { n: f.source_sessions, label: 'FULL DATA', desc: 'Diginetica 原始点击日志；' + fmt(f.source_items) + ' 个商品。', w: 100 },
      { n: f.eligible_sessions, label: 'AFTER 5-CORE', desc: '完整数据先做 item 5-core，再排除 smoke 会话。', w: Math.max(8, f.eligible_sessions / max * 100) },
      { n: f.selected_sessions, label: 'SAMPLED', desc: '固定 seed=2026，抽取 5–20 次点击的完整会话。', w: Math.max(8, f.selected_sessions / max * 100) },
      { n: f.selected_sessions, label: 'TEST SET', desc: '会话级留一划分；每会话一个真实“下一次点击”。', w: Math.max(8, f.selected_sessions / max * 100) }
    ];
    $('#funnel').innerHTML = steps.map(s => `
      <div class="funnel-step">
        <div class="fnum">${fmt(s.n)}</div>
        <div class="flabel">${s.label}</div>
        <div class="fdesc">${s.desc}</div>
        <div class="fbar"><span style="--w:${s.w}%"></span></div>
      </div>`).join('');

    const m = DATA.meta;
    const kpis = [
      [fmt(m.counts.test), '测试会话'],
      [fmt(m.counts.train), '训练样本'],
      [fmt(m.counts.items), '候选商品'],
      [m.epochs + ' 轮 · seed ' + m.seed, '训练设置']
    ];
    $('#kpiRow').innerHTML = kpis.map(([v, l]) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
  }

  /* ───────────── 02 对照 ───────────── */
  function initCompare() {
    const metrics = DATA.metrics;
    const times = metrics.map(m => m.time);
    const recalls = metrics.map(m => m.recall * 100);
    const params = metrics.map(m => m.params);
    const minP = Math.min(...params), maxP = Math.max(...params);
    const maxRecall = Math.max(...recalls);
    const minTime = Math.min(...times), maxTime = Math.max(...times);

    const chart = makeChart($('#qualityCost'), 'qualityCost');
    chart.setOption(Object.assign(baseOption(), {
      grid: { left: 58, right: 42, top: 46, bottom: 58, containLabel: true },
      xAxis: {
        type: 'value', name: '训练 + 验证时间（秒）', nameLocation: 'middle', nameGap: 34,
        nameTextStyle: { color: C.muted, fontSize: 10 },
        min: Math.max(0, Math.floor(minTime * 0.7 / 10) * 10),
        max: Math.ceil(maxTime * 1.18 / 10) * 10,
        axisLine: { lineStyle: { color: '#B9AD97' } },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 10 }
      },
      yAxis: {
        type: 'value', name: 'Recall@10（%）', nameLocation: 'middle', nameGap: 46, nameRotate: 90,
        nameTextStyle: { color: C.muted, fontSize: 10 },
        min: 0, max: Math.ceil(maxRecall * 1.25 / 5) * 5,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 10 }
      },
      series: [{
        type: 'scatter',
        data: metrics.map((m, i) => ({
          value: [m.time, m.recall * 100],
          name: LABELS[m.key],
          symbolSize: 20 + (params[i] - minP) / Math.max(1, maxP - minP) * 22,
          itemStyle: { color: MODEL_COLOR[m.key], borderColor: C.paper, borderWidth: 2 },
          label: { show: true, formatter: LABELS[m.key], position: 'top', color: C.ink, fontSize: 11, fontWeight: 600 }
        })),
        emphasis: { scale: 1.12 },
        tooltip: {
          formatter: (p) => {
            const m = metrics[p.dataIndex];
            return `<b>${LABELS[m.key]}</b><br/>Recall@10 ${pct(m.recall)}<br/>MRR@10 ${pct(m.mrr)}<br/>NDCG@10 ${pct(m.ndcg)}<br/>训练 ${m.time.toFixed(1)} 秒<br/>参数 ${fmt(m.params)}`;
          }
        }
      }],
      graphic: [
        { type: 'text', left: 70, top: 22, style: { text: '↑ 更准', fill: C.muted, font: '10px "Cascadia Code", monospace' } },
        { type: 'text', right: 34, top: 22, style: { text: '更贵 →', fill: C.muted, font: '10px "Cascadia Code", monospace' } }
      ]
    }), true);

    const bar = makeChart($('#metricBars'), 'metricBars');
    bar.setOption(Object.assign(baseOption(), {
      grid: { left: 62, right: 24, top: 50, bottom: 34, containLabel: true },
      legend: {
        top: 4, right: 0, itemWidth: 9, itemHeight: 9, itemGap: 14,
        textStyle: { color: C.ink2, fontSize: 10 }
      },
      xAxis: {
        type: 'category', data: metrics.map(m => LABELS[m.key]),
        axisLine: { lineStyle: { color: '#B9AD97' } }, axisTick: { show: false },
        axisLabel: { color: C.ink2, fontSize: 11, fontFamily: '"Cascadia Code", monospace' }
      },
      yAxis: {
        type: 'value', name: '占比（%）', nameLocation: 'middle', nameGap: 40, nameRotate: 90,
        nameTextStyle: { color: C.muted, fontSize: 10 },
        max: Math.ceil(Math.max(...metrics.map(m => Math.max(m.recall, m.mrr, m.ndcg))) * 125 / 10) * 10,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 10 }
      },
      series: ['recall', 'mrr', 'ndcg'].map((key, i) => ({
        name: key === 'recall' ? 'Recall@10' : key === 'mrr' ? 'MRR@10' : 'NDCG@10',
        type: 'bar', barGap: '12%', barWidth: '18%',
        itemStyle: { color: [C.ink, C.teal, C.gold][i] },
        data: metrics.map(m => +(m[key] * 100).toFixed(2)),
        label: { show: true, position: 'top', fontSize: 9, color: C.muted, formatter: (p) => p.value.toFixed(1) }
      }))
    }), true);

    const best = metrics.find(m => m.model === DATA.summary.best_model) || metrics.reduce((a, b) => a.recall >= b.recall ? a : b);
    const srgnn = metrics.find(m => m.key === 'SRGNN');
    const gain = ((best.recall - srgnn.recall) / srgnn.recall * 100).toFixed(1);
    const dt = (best.time - srgnn.time).toFixed(1);
    const ratio = (best.time / srgnn.time).toFixed(2);
    $('#compareQuote').innerHTML = `${LABELS[best.key]} 的 Recall@10 比 SR-GNN 高 <strong>${gain}%</strong>，训练时间多 <strong>${dt} 秒</strong>（${ratio}×）。`;
  }

  /* ───────────── 03 学习过程 ───────────── */
  function initLearning() {
    const models = DATA.metrics.map(m => m.key);
    const maxEpoch = Math.max(...models.map(k => (DATA.curves[k] || []).length), 1);
    const epochs = Array.from({ length: maxEpoch }, (_, i) => i + 1);

    const mrr = makeChart($('#mrrCurve'), 'mrrCurve');
    const mrrSeries = models.map(k => ({
      name: LABELS[k], type: 'line', smooth: false, symbol: 'circle', symbolSize: 5,
      lineStyle: { width: 2, color: MODEL_COLOR[k] },
      itemStyle: { color: MODEL_COLOR[k] },
      data: epochs.map(e => {
        const row = (DATA.curves[k] || []).find(x => x.epoch === e);
        return row && row.valid_mrr != null ? +(row.valid_mrr * 100).toFixed(2) : null;
      })
    }));
    const maxMrr = Math.max(...mrrSeries.flatMap(s => s.data.filter(v => v != null)), 1);
    mrr.setOption(Object.assign(baseOption(), {
      grid: { left: 52, right: 24, top: 42, bottom: 40, containLabel: true },
      legend: { top: 0, right: 0, itemWidth: 18, itemHeight: 2, textStyle: { color: C.ink2, fontSize: 10 } },
      xAxis: {
        type: 'category', data: epochs, name: '轮次', nameTextStyle: { color: C.muted, fontSize: 9 },
        axisLine: { lineStyle: { color: '#B9AD97' } }, axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: 9 }
      },
      yAxis: {
        type: 'value', name: 'MRR@10（%）', nameLocation: 'middle', nameGap: 44, nameRotate: 90,
        nameTextStyle: { color: C.muted, fontSize: 9 },
        min: 0, max: Math.ceil(maxMrr * 1.18 / 5) * 5,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 9 }
      },
      series: mrrSeries
    }), true);

    const loss = makeChart($('#lossCurve'), 'lossCurve');
    const lossSeries = models.map(k => ({
      name: LABELS[k], type: 'line', smooth: false, symbol: 'none',
      lineStyle: { width: 2, color: MODEL_COLOR[k] },
      data: epochs.map(e => {
        const row = (DATA.curves[k] || []).find(x => x.epoch === e);
        return row && row.loss > 0 ? +row.loss.toFixed(2) : null;
      })
    }));
    loss.setOption(Object.assign(baseOption(), {
      grid: { left: 62, right: 24, top: 42, bottom: 40, containLabel: true },
      legend: { top: 0, right: 0, itemWidth: 18, itemHeight: 2, textStyle: { color: C.ink2, fontSize: 10 } },
      xAxis: {
        type: 'category', data: epochs, name: '轮次', nameTextStyle: { color: C.muted, fontSize: 9 },
        axisLine: { lineStyle: { color: '#B9AD97' } }, axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: 9 }
      },
      yAxis: {
        type: 'log', name: 'loss（对数轴）', nameLocation: 'middle', nameGap: 50, nameRotate: 90,
        nameTextStyle: { color: C.muted, fontSize: 9 },
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 9 }
      },
      series: lossSeries
    }), true);

    const best = DATA.metrics.find(m => m.model === DATA.summary.best_model) || DATA.metrics[0];
    const rows = (DATA.curves[best.key] || []).filter(x => x.valid_mrr != null);
    if (rows.length) {
      const top = rows.reduce((a, b) => a.valid_mrr >= b.valid_mrr ? a : b);
      $('#learningNote').textContent = `${LABELS[best.key]} 在第 ${top.epoch} 轮取得最高验证 MRR@10（${pct(top.valid_mrr)}）。曲线只说明本轮训练过程，不据此宣称充分收敛。`;
    } else {
      $('#learningNote').textContent = '训练日志中没有可解析的验证曲线。';
    }
  }

  /* ───────────── 04 回放 ───────────── */
  const KIND_LABEL = { success: '成功', failure: '失败', disagreement: '分歧' };

  function caseLabel(c, i) {
    return `#${String(i + 1).padStart(2, '0')} · ${c.id} · ${KIND_LABEL[c.kind] || ''} · 长度 ${c.history_len}`;
  }

  function initReplay() {
    const select = $('#caseSelect');
    select.innerHTML = DATA.cases.map((c, i) => `<option value="${i}">${caseLabel(c, i)}</option>`).join('');
    select.addEventListener('change', () => { state.caseIndex = +select.value; renderCase(); });
    $('#prevCase').addEventListener('click', () => { state.caseIndex = (state.caseIndex - 1 + DATA.cases.length) % DATA.cases.length; renderCase(); });
    $('#nextCase').addEventListener('click', () => { state.caseIndex = (state.caseIndex + 1) % DATA.cases.length; renderCase(); });
    document.querySelectorAll('#modelTabs button').forEach(btn => {
      btn.addEventListener('click', () => {
        state.model = btn.dataset.model;
        document.querySelectorAll('#modelTabs button').forEach(b => b.classList.toggle('active', b === btn));
        renderCase();
      });
    });
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowLeft') { state.caseIndex = (state.caseIndex - 1 + DATA.cases.length) % DATA.cases.length; renderCase(); }
      if (e.key === 'ArrowRight') { state.caseIndex = (state.caseIndex + 1) % DATA.cases.length; renderCase(); }
    });
    renderCase();
  }

  function historyFrames(c) {
    const h = c.history;
    const seen = new Set();
    const frames = [];
    const add = (item, idx, isTarget) => {
      const dup = !isTarget && seen.has(item);
      if (!isTarget) seen.add(item);
      frames.push(`<div class="frame ${dup ? 'dup' : ''} ${isTarget ? 'target' : ''}">
        <span class="idx">${isTarget ? 'NEXT' : String(idx + 1).padStart(2, '0')}</span>
        <span class="item">${item}</span>
        ${dup ? '<span class="tag">重复</span>' : ''}
      </div>`);
    };
    if (h.length <= 7) {
      h.forEach((x, i) => add(x, i, false));
    } else {
      h.slice(0, 3).forEach((x, i) => add(x, i, false));
      frames.push('<div class="history-arrow">···</div>');
      h.slice(-3).forEach((x, i) => add(x, h.length - 3 + i, false));
    }
    frames.push('<div class="history-arrow">→</div>');
    add(c.target, h.length, true);
    return frames.join('');
  }

  function renderCase() {
    const c = DATA.cases[state.caseIndex];
    if (!c) return;
    const modelData = c.models[state.model];
    const recs = modelData.recommendations;
    const rank = modelData.rank;

    $('#caseSelect').value = String(state.caseIndex);
    $('#caseMeta').innerHTML = [
      `会话 ${c.id}`,
      `历史 ${c.history_len} 次`,
      `重复点击 ${c.repeat_count}`,
      `目标热度 #${c.target_popularity_rank || '—'}`
    ].map(x => `<span>${x}</span>`).join('');
    $('#historyStrip').innerHTML = historyFrames(c);
    $('#historyStrip').title = c.history.join(' → ');

    const verdict = $('#verdict');
    verdict.className = 'verdict ' + (rank === 0 ? 'miss' : rank === 1 ? 'hit' : 'rank');
    verdict.innerHTML = rank === 0
      ? `未命中 Top-10 <small>目标 ${c.target} 不在前十</small>`
      : rank === 1
        ? `命中 · 第 1 位 <small>目标 ${c.target}</small>`
        : `命中 · 第 ${rank} 位 <small>目标 ${c.target}</small>`;

    $('#replaySubtitle').textContent = `${LABELS[state.model]} · 真实下一商品 ${c.target}`;

    const chart = makeChart($('#replayChart'), 'replay');
    const ordered = recs.slice().reverse();
    const items = ordered.map(r => `#${r.rank}  ${r.item}`);
    const values = ordered.map(r => r.score);
    const colors = ordered.map(r => r.item === c.target ? C.green : (r.rank <= 3 ? C.teal : '#B7B0A2'));
    chart.setOption(Object.assign(baseOption(), {
      grid: { left: 116, right: 70, top: 14, bottom: 36, containLabel: false },
      xAxis: {
        type: 'value', name: '模型得分', nameLocation: 'middle', nameGap: 24,
        nameTextStyle: { color: C.muted, fontSize: 9 },
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: '#E6DCCC', type: 'dashed' } },
        axisLabel: { color: C.muted, fontSize: 9 }
      },
      yAxis: {
        type: 'category', data: items,
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: {
          color: (value, index) => ordered[index] && ordered[index].item === c.target ? C.green : C.ink2,
          fontSize: 10, fontFamily: '"Cascadia Code", monospace'
        }
      },
      series: [{
        type: 'bar', barWidth: '58%',
        data: values.map((v, i) => ({ value: v, itemStyle: { color: colors[i] } })),
        label: { show: true, position: 'right', fontSize: 9, color: C.muted, formatter: (p) => p.value.toFixed(3) },
        tooltip: {
          formatter: (p) => {
            const r = ordered[p.dataIndex];
            return `<b>#${r.rank} · ${r.item}</b><br/>得分 ${r.score.toFixed(4)}${r.item === c.target ? '<br/><span style="color:#8FD3A8">真实下一商品</span>' : ''}`;
          }
        }
      }]
    }), true);
    renderGraph();
  }

  /* ───────────── 05 图 ───────────── */
  function initGraph() {
    if (new URLSearchParams(location.search).get('graph') === 'global') state.graphMode = 'global';
    document.querySelectorAll('#graphToggle button').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === state.graphMode);
      btn.addEventListener('click', () => {
        state.graphMode = btn.dataset.mode;
        document.querySelectorAll('#graphToggle button').forEach(b => b.classList.toggle('active', b === btn));
        renderGraph();
      });
    });
    renderGraph();
  }

  function renderGraph() {
    const c = DATA.cases[state.caseIndex];
    if (!c) return;
    const chart = makeChart($('#graphChart'), 'graph');
    const local = state.graphMode === 'local';

    if (local) {
      $('#graphHint').textContent = '节点大小 = 会话内出现次数；箭头 = 连续点击';
      const nodes = c.graph.nodes;
      const nodeData = nodes.map(n => ({
        name: n.id,
        value: n.count,
        symbolSize: 16 + Math.min(n.count, 5) * 7,
        itemStyle: { color: n.id === c.target ? C.vermillion : C.teal, borderColor: C.paper, borderWidth: 2 },
        label: { show: nodes.length <= 14 || n.count > 1 }
      }));
      const links = c.graph.edges.map(e => ({
        source: e.source, target: e.target, value: e.value,
        lineStyle: { width: 1 + Math.min(e.value, 3) * 0.9, curveness: 0.07, opacity: 0.55 }
      }));
      chart.setOption({
        textStyle: { fontFamily: '"Microsoft YaHei", sans-serif' },
        tooltip: Object.assign(baseOption().tooltip, { formatter: (p) => p.dataType === 'edge' ? `连续点击 ${p.data.value} 次` : `${p.name}<br/>出现 ${p.value} 次` }),
        series: [{
          type: 'graph', layout: 'force', roam: true, draggable: true,
          data: nodeData, links: links,
          force: { repulsion: 160, edgeLength: [60, 120], gravity: 0.08 },
          label: { color: C.ink2, fontSize: 10, fontFamily: '"Cascadia Code", monospace', position: 'bottom', distance: 4 },
          edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 5,
          lineStyle: { color: '#9A9488' },
          emphasis: { focus: 'adjacency', lineStyle: { width: 3 } }
        }]
      }, true);
    } else {
      $('#graphHint').textContent = '中心 = 会话最后一个商品；外圈 = 训练集共现邻居；边宽 = 共现权重';
      const last = c.history[c.history.length - 1];
      const seen = new Set();
      const neighbours = (c.global_neighbours || []).filter(n => {
        if (n.id === last || seen.has(n.id)) return false;
        seen.add(n.id);
        return true;
      });
      if (!neighbours.length) {
        chart.clear();
        chart.setOption({
          graphic: [{
            type: 'text', left: 'center', top: 'middle',
            style: { text: '该商品在训练集中没有可展示的全局邻居', fill: C.muted, fontSize: 13 }
          }]
        });
        return;
      }
      const maxW = Math.max(...neighbours.map(n => n.weight), 1);
      const nodes = [{ name: last, x: 0, y: 0, symbolSize: 54, itemStyle: { color: C.vermillion, borderColor: C.paper, borderWidth: 2 }, label: { show: true, fontWeight: 600 } }];
      const links = [];
      neighbours.slice(0, 12).forEach((n, i, arr) => {
        const angle = (Math.PI * 2 * i) / arr.length - Math.PI / 2;
        const r = 190;
        nodes.push({
          name: n.id, x: Math.cos(angle) * r, y: Math.sin(angle) * r,
          symbolSize: 12 + (n.weight / maxW) * 16,
          itemStyle: { color: n.id === c.target ? C.green : C.teal, borderColor: C.paper, borderWidth: 1.5 },
          label: { show: true, position: 'right', fontSize: 9 }
        });
        links.push({ source: last, target: n.id, value: n.weight, lineStyle: { width: 1 + (n.weight / maxW) * 3, opacity: 0.5 } });
      });
      chart.setOption({
        textStyle: { fontFamily: '"Microsoft YaHei", sans-serif' },
        tooltip: Object.assign(baseOption().tooltip, { formatter: (p) => p.dataType === 'edge' ? `共现权重 ${p.data.value}` : `${p.name}` }),
        series: [{
          type: 'graph', layout: 'none', roam: true, data: nodes, links: links,
          label: { color: C.ink2, fontFamily: '"Cascadia Code", monospace' },
          edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 4,
          lineStyle: { color: '#9A9488' }
        }]
      }, true);
    }
  }

  /* ───────────── 06 失败 ───────────── */
  function initFailures() {
    let fails = DATA.cases.filter(c => c.kind === 'failure');
    if (!fails.length) fails = DATA.cases.filter(c => c.kind === 'disagreement');
    fails = fails.slice(0, 3);
    if (!fails.length) {
      $('#failureGrid').innerHTML = '<div class="note"><div class="note-mark">i</div><p>本次精选会话中没有失败案例。</p></div>';
      return;
    }
    $('#failureGrid').innerHTML = fails.map(c => {
      const h = c.history;
      const hist = h.length <= 7 ? h.join(' → ') : h.slice(0, 3).join(' → ') + ' → ··· → ' + h.slice(-2).join(' → ');
      const rows = ['SRGNN', 'GCSAN', 'GCEGNN'].map(k => {
        const rank = c.models[k].rank;
        const w = rank === 0 ? 100 : Math.max(8, (10 - rank) / 10 * 100);
        return `<div class="failure-row ${rank === 0 ? 'miss' : 'hit'}">
          <span class="failure-model">${LABELS[k]}</span>
          <div class="bar"><span style="--w:${w}%"></span></div>
          <em>${rank === 0 ? '未命中' : '第 ' + rank + ' 位'}</em>
        </div>`;
      }).join('');
      return `<div class="failure-card">
        <div class="failure-head"><b>会话 ${c.id}</b><span>FAILURE CASE</span></div>
        <div class="failure-history">${hist}</div>
        <div class="failure-target">真实下一商品 <b>${c.target}</b> · 历史 ${c.history_len} 次 · 重复 ${c.repeat_count}</div>
        <div class="failure-ranks">${rows}</div>
      </div>`;
    }).join('');
  }

  /* ───────────── 导航 / 进度 / 入场 ───────────── */
  function initNav() {
    const links = Array.from(document.querySelectorAll('#railNav a'));
    const sections = links.map(a => document.getElementById(a.dataset.section)).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          links.forEach(a => a.classList.toggle('active', a.dataset.section === entry.target.id));
        }
      });
    }, { rootMargin: '-35% 0px -55% 0px', threshold: 0 });
    sections.forEach(s => observer.observe(s));

    const bar = $('#progressBar');
    const update = () => {
      const doc = document.documentElement;
      const total = doc.scrollHeight - window.innerHeight;
      bar.style.width = (total > 0 ? clamp(window.scrollY / total, 0, 1) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', update, { passive: true });
    update();

    const revealTargets = document.querySelectorAll('.section-head, .funnel, .kpi-row, .note-row, .compare-grid, .pull, .two-col, .replay-stage, .graph-stage, .failure-grid, .limits-grid');
    revealTargets.forEach(el => el.classList.add('reveal'));
    const reveal = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          reveal.unobserve(entry.target);
        }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    revealTargets.forEach(el => reveal.observe(el));
  }

  function initResize() {
    let raf = null;
    window.addEventListener('resize', () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => charts.forEach(c => c.resize()));
    });
  }

  function initDeepLink() {
    const only = window.__ONLY__ || new URLSearchParams(location.search).get('only');
    if (only) {
      document.querySelectorAll('main > section').forEach(s => {
        if (s.id !== only) s.style.display = 'none';
      });
      setTimeout(() => charts.forEach(c => c.resize()), 120);
      window.scrollTo(0, 0);
      return;
    }
    if (!location.hash) return;
    const el = document.querySelector(location.hash);
    if (el) setTimeout(() => el.scrollIntoView({ behavior: 'auto', block: 'start' }), 150);
  }

  function showError(message) {
    document.body.innerHTML = '<div style="padding:48px;font-family:Microsoft YaHei;color:#16272B"><h1>数据没有加载</h1><p>' + message + '</p><p>请确认 web/data.js 已生成，并通过 index.html 打开。</p></div>';
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (!DATA) { showError('window.__DATA__ 为空。'); return; }
    try {
      initCover();
      initOverview();
      initCompare();
      initLearning();
      initReplay();
      initGraph();
      initFailures();
      initNav();
      initResize();
      initDeepLink();
    } catch (err) {
      console.error(err);
      showError(String(err && err.message || err));
    }
  });
})();
