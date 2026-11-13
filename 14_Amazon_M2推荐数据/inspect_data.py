"""Inspect the two original CSV files without creating training subsets."""
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
csv.field_size_limit(64 * 1024 * 1024)
ASIN_TOKEN = re.compile(r'''['"]([A-Za-z0-9]{10})['"]''')


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def history(row):
    raw = row['prev_items']
    # Release files may use NumPy-style adjacent quoted ASINs without commas.
    # literal_eval would silently concatenate adjacent string literals.
    value = ASIN_TOKEN.findall(raw)
    residue = ASIN_TOKEN.sub('', raw)
    if re.sub(r'[\s,\[\]]', '', residue):
        raise ValueError('Unexpected history encoding: ' + raw[:120])
    return value


def main():
    session_path = RAW / 'sessions_train.csv'
    product_path = RAW / 'products_train.csv'
    samples = {}
    with session_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        session_columns = reader.fieldnames
        for row in reader:
            locale = row['locale']
            h = history(row)
            if locale not in samples and len(h) >= 2:
                samples[locale] = {'locale': locale, 'history': h, 'target': row['next_item']}
            if len(samples) == 6:
                break
    wanted = {(s['locale'], item) for s in samples.values() for item in s['history'] + [s['target']]}
    names = {}
    catalogue = defaultdict(set)
    product_counts = Counter()
    coverage = defaultdict(Counter)
    duplicate_keys = 0
    positive_prices = Counter()
    with product_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        product_columns = reader.fieldnames
        for row in reader:
            locale, item = row['locale'], row['id']
            duplicate_keys += item in catalogue[locale]
            catalogue[locale].add(item)
            product_counts[locale] += 1
            for field in product_columns:
                if row[field].strip():
                    coverage[locale][field] += 1
            try:
                price = float(row['price'])
                positive_prices[locale] += math.isfinite(price) and price > 0
            except ValueError:
                pass
            if (locale, item) in wanted:
                names[(locale, item)] = {'id': item, 'title': row['title'],
                                         'brand': row['brand'], 'price': row['price']}
    print('Products checked', dict(product_counts), flush=True)
    session_counts = Counter()
    interactions = Counter()
    lengths = defaultdict(Counter)
    repeat_sessions = Counter()
    target_in_history = Counter()
    missing_history_refs = Counter()
    missing_target_refs = Counter()
    used = defaultdict(set)
    total = 0
    with session_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            locale, target = row['locale'], row['next_item']
            h = history(row)
            sequence = h + [target]
            session_counts[locale] += 1
            interactions[locale] += len(sequence)
            lengths[locale][len(h)] += 1
            repeat_sessions[locale] += len(set(sequence)) < len(sequence)
            target_in_history[locale] += target in h
            missing_history_refs[locale] += sum(item not in catalogue[locale] for item in h)
            missing_target_refs[locale] += target not in catalogue[locale]
            used[locale].update(sequence)
            total += 1
            if total % 500000 == 0:
                print(f'Sessions checked {total:,}', flush=True)
    summaries = {}
    for locale in sorted(session_counts):
        count = session_counts[locale]
        summaries[locale] = {
            'sessions': count,
            'product_metadata_rows': product_counts[locale],
            'unique_interacted_products': len(used[locale]),
            'interactions_including_targets': interactions[locale],
            'mean_history_length': sum(n*c for n,c in lengths[locale].items()) / count,
            'min_history_length': min(lengths[locale]),
            'max_history_length': max(lengths[locale]),
            'sessions_with_repeats_including_target': repeat_sessions[locale],
            'target_already_in_history': target_in_history[locale],
            'missing_history_product_references': missing_history_refs[locale],
            'missing_target_product_references': missing_target_refs[locale],
            'product_nonempty_fields': dict(coverage[locale]),
            'finite_positive_prices': positive_prices[locale],
        }
    result = {'session_columns': session_columns, 'product_columns': product_columns,
              'total_sessions': total, 'total_product_rows': sum(product_counts.values()),
              'total_history_interactions': sum(interactions.values()) - total,
              'total_target_labels': total,
              'target_already_in_history': sum(target_in_history.values()),
              'duplicate_locale_product_keys': duplicate_keys,
              'locales': summaries,
              'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                        for p in [session_path, product_path]},
              'history_length_distribution': {loc: dict(sorted(c.items())) for loc,c in lengths.items()}}
    save('数据核查.json', result)
    examples = []
    for locale in sorted(samples, key=lambda loc: (loc != 'UK', loc)):
        s = samples[locale]
        examples.append({'locale': locale,
                         'history': [names.get((locale,item), {'id':item, 'title':'[missing]'}) for item in s['history']],
                         'target': names.get((locale,s['target']), {'id':s['target'], 'title':'[missing]'})})
    save('真实会话样例.json', examples)
    lines = ['# Amazon-M2 数据说明', '',
             '已下载Kaggle公开副本，并提取原始训练会话和商品属性两个CSV；本目录保留未经抽样的原始数据，模型实验在独立目录开展。', '',
             '原始来源：[KDD Cup 2023官方数据介绍](https://kddcup23.github.io/)。',
             '本次下载：[Kaggle公开副本](https://www.kaggle.com/datasets/marquis03/amazon-m2)，版本1。', '',
             '## 1. 实际数据规模', '',
             '| 站点 | 训练会话 | 属性表商品行数 | 会话涉及商品 | 序列项数（含目标） | 平均历史长度 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for loc,s in summaries.items():
        lines.append(f'| {loc} | {s["sessions"]:,} | {s["product_metadata_rows"]:,} | '
                     f'{s["unique_interacted_products"]:,} | {s["interactions_including_targets"]:,} | '
                     f'{s["mean_history_length"]:.2f} |')
    lines += ['', f'合计 **{total:,}条训练会话、{sum(product_counts.values()):,}行商品属性**。商品以`locale + id`关联，同一ASIN可能出现在多个站点，因此属性行数不等于跨站点去重商品数。', '',
              f'历史商品交互共 **{sum(interactions.values())-total:,}项**，另有 **{total:,}个下一商品标签**。表中的序列项数包含标签，与只统计prev_items的历史交互数口径不同。', '',
              'UK为英国、DE为德国、JP为日本、ES为西班牙、FR为法国、IT为意大利。上表“训练会话”指发布方的训练文件，并非本地已经完成训练。', '',
              '## 2. 两个文件分别是什么', '',
              '| 文件 | 实际大小（十进制MB） | 内容 |',
              '| --- | ---: | --- |',
              f'| `products_train.csv` | {product_path.stat().st_size/1e6:.2f} | 商品目录与可读属性 |',
              f'| `sessions_train.csv` | {session_path.stat().st_size/1e6:.2f} | 每行一个历史商品序列及下一商品标签 |', '',
              '`sessions_train.csv`实际列：' + '、'.join(f'`{c}`' for c in session_columns) + '。', '',
              '- `prev_items`：该会话此前交互的商品ID列表，按发布方给定的时间顺序排列。',
              '- `next_item`：下一件交互商品，是训练/评价标签，不可混入输入历史。',
              '- `locale`：站点，用于匹配同站点商品属性。', '',
              '`products_train.csv`实际列：' + '、'.join(f'`{c}`' for c in product_columns) + '。', '',
              '属性包括标题、价格、品牌、颜色、尺寸、型号、材质、作者和描述；不同商品会有缺失。价格需按站点解释，跨站点不能直接比较数值。', '',
              '## 3. 推荐任务与真实样例', '',
              '任务：`prev_items + 历史/候选商品属性 → 给候选商品打分 → 预测next_item`。它来自真实购物会话，区别于评论记录重建的跨日用户序列。官方描述每个会话为30分钟活动窗口。', '',
              '以下直接读取英国站的一条记录，并按商品表映射标题。保留原始英文，未人工替换商品：', '']
    sample = examples[0]
    for i,item in enumerate(sample['history'],1):
        lines.append(f'{i}. 已交互：**{item["title"]}**（`{item["id"]}`）')
    lines += ['', f'真实下一商品：**{sample["target"]["title"]}**（`{sample["target"]["id"]}`）。', '',
              '这里展示的是数据标签，不是模型已经推荐成功。其他站点样例见 `真实会话样例.json`。', '',
              '## 4. 后续实验必须保持的条件', '',
              '- GNN可使用序列转移图；灰关联可组合行为关联及价格、品牌等属性；离线Qwen可基于可读标题与结构化属性分析策略。可运行性及效果仍需后续实验确认。',
              '- 原始数据按完整会话保留。本次没有再次只截取2000个用户训练；后续实验集应在足够训练信息的基础上另行划分。',
              '- 会话表没有提供用户ID或逐次点击时间戳，不能从这两个文件恢复跨会话用户身份，也不能计算真实点击间隔。可使用给定的商品顺序。',
              f'- 本次逐行核查，下一商品已在历史中出现的会话为 **{sum(target_in_history.values()):,}条**；历史序列自身存在重复商品的会话为 **{sum(repeat_sessions.values()):,}条**。保留这些重复节点及转移，不把原始历史简单去重。目标过滤规则应随新实验协议统一确定。',
              '- 商品属性为提供的目录快照；目标标签不得参与特征构造。词表、统计与超参数分别按训练/验证划分处理。',
              '- 官方任务1是下一商品推荐；跨站点迁移和标题生成是不同任务。本次未解压这些额外任务文件。', '',
              '## 5. 本地文件', '',
              '- `data/raw/amazon-m2-v1.zip`：原始下载包，保留未使用的任务文件在压缩包内。',
              '- `data/raw/products_train.csv`、`data/raw/sessions_train.csv`：本次实际解压文件。',
              '- `下载清单.json`：下载来源、ZIP目录、大小、完整性及哈希。',
              '- `数据核查.json`：逐行统计、属性非空数、目录匹配及文件哈希。',
              '- `真实会话样例.json`：六个站点的真实历史及目标商品标题。',
              '- `inspect_data.py`：可重复执行的流式核查脚本。', '']
    (ROOT / '数据说明.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'sessions': total, 'products': sum(product_counts.values()),
                      'duplicate_keys': duplicate_keys,
                      'missing_target_refs': sum(missing_target_refs.values()),
                      'missing_history_refs': sum(missing_history_refs.values())}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
