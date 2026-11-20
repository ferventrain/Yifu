window.__ModuleLoader__.load({
  id: '@local/yifu-pm',
  factory(require) {
    const React = require('react')
    const h = React.createElement
    const POLL_MS = 2000

    // ---------------------------------------------------------------- theme
    const COLORS = {
      running: 'var(--color-blue-500, #4aa8ff)',
      queued: 'var(--color-fg-tertiary, #8a94a6)',
      verifying: 'var(--color-teal-500, #14b8a6)',
      succeeded: 'var(--color-green-500, #22c55e)',
      failed: 'var(--color-red-500, #ef4444)',
      cancelled: 'var(--color-orange-500, #f59e0b)',
      stalled: 'var(--color-yellow-500, #eab308)',
    }
    const LABELS = {
      running: '运行中',
      queued: '排队',
      verifying: '校验中',
      succeeded: '成功',
      failed: '失败',
      cancelled: '已取消',
      stalled: '失联',
    }
    const MUTED = 'var(--color-fg-tertiary, #8a94a6)'
    const BORDER = '1px solid var(--color-border-primary, rgba(127,127,127,.18))'
    const SOFT_BG = 'var(--color-bg-secondary, rgba(127,127,127,.06))'
    const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

    // ------------------------------------------------------------- format
    function fmtDuration(seconds) {
      if (seconds === null || seconds === undefined) return null
      const total = Math.max(0, Math.round(Number(seconds)))
      const hours = Math.floor(total / 3600)
      const minutes = Math.floor((total % 3600) / 60)
      if (hours > 0) return `${hours}h ${String(minutes).padStart(2, '0')}m`
      if (minutes > 0) return `${minutes}m ${String(total % 60).padStart(2, '0')}s`
      return `${total}s`
    }

    function fmtEta(seconds) {
      const text = fmtDuration(seconds)
      return text === null ? null : `≈${text}`
    }

    function fmtClock(iso) {
      if (!iso) return '—'
      const date = new Date(iso)
      if (Number.isNaN(date.getTime())) return '—'
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    function fmtAgo(seconds) {
      if (seconds === null || seconds === undefined) return ''
      const value = Number(seconds)
      if (value < 60) return `${Math.round(value)} 秒前`
      if (value < 3600) return `${Math.round(value / 60)} 分钟前`
      return `${Math.round(value / 3600)} 小时前`
    }

    // ---------------------------------------------------------- primitives
    function statusPill(status) {
      const color = COLORS[status] || MUTED
      return h(
        'span',
        {
          style: {
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 11.5,
            padding: '2px 8px',
            borderRadius: 999,
            color,
            border: `1px solid color-mix(in srgb, ${color} 35%, transparent)`,
            background: `color-mix(in srgb, ${color} 10%, transparent)`,
            whiteSpace: 'nowrap',
          },
        },
        h('span', { style: { width: 6, height: 6, borderRadius: '50%', background: color } }),
        LABELS[status] || status || '—',
      )
    }

    function stepDots(row) {
      const total = Number(row.step_total)
      const index = Number(row.step_index) || 0
      if (!Number.isFinite(total) || total <= 0 || total > 12) return null
      const dots = []
      for (let i = 1; i <= total; i += 1) {
        const failed = (row.display === 'failed' || row.display === 'stalled') && i === Math.max(index, 1)
        const done = !failed && (row.display === 'succeeded' || (index > 0 && i < index))
        const active = row.display === 'running' && i === index
        dots.push(
          h('span', {
            key: i,
            title: `step ${i}/${total}`,
            style: {
              width: active ? 16 : 8,
              height: 6,
              borderRadius: 4,
              background: failed ? COLORS.failed : done ? COLORS.succeeded : active ? COLORS.running : 'var(--color-bg-tertiary, rgba(127,127,127,.22))',
              transition: 'width .2s ease, background .2s ease',
            },
          }),
        )
      }
      return h('div', { style: { display: 'flex', alignItems: 'center', gap: 3 } }, dots)
    }

    function progressBar(row) {
      const percent = row.percent
      const color = COLORS[row.display] || COLORS.running
      return h(
        'div',
        { style: { display: 'flex', alignItems: 'center', gap: 8 } },
        h(
          'div',
          {
            style: {
              position: 'relative',
              flex: '1 1 auto',
              height: 5,
              borderRadius: 3,
              background: 'var(--color-bg-tertiary, rgba(127,127,127,.2))',
              overflow: 'hidden',
            },
          },
          percent === null
            ? h('div', {
                style: {
                  position: 'absolute',
                  inset: 0,
                  background:
                    'repeating-linear-gradient(90deg, var(--color-bg-tertiary, rgba(127,127,127,.28)) 0 6px, transparent 6px 12px)',
                },
              })
            : h('div', {
                style: {
                  position: 'absolute',
                  inset: 0,
                  width: `${percent}%`,
                  background: color,
                  transition: 'width .3s ease',
                },
              }),
        ),
        h(
          'span',
          { style: { fontSize: 11, color: MUTED, minWidth: 40, textAlign: 'right', fontVariantNumeric: 'tabular-nums' } },
          percent === null ? '未知' : `${percent}%`,
        ),
      )
    }

    function errorBadge(error) {
      if (!error) return null
      return h(
        'span',
        {
          title: error.message || '',
          style: {
            fontSize: 10.5,
            padding: '1px 6px',
            borderRadius: 4,
            color: 'var(--color-red-500, #ef4444)',
            background: 'color-mix(in srgb, var(--color-red-500, #ef4444) 12%, transparent)',
            whiteSpace: 'nowrap',
          },
        },
        error.code || 'ERROR',
      )
    }

    // ---------------------------------------------------------------- rows
    function jobRow(row, selected, onSelect) {
      const isSelected = selected === row.run_id
      const running = row.display === 'running'
      const duration = fmtDuration(row.duration_s)
      const eta = row.display === 'running' ? fmtEta(row.eta_total_s) : null
      return h(
        'div',
        {
          key: row.run_id,
          role: 'button',
          tabIndex: 0,
          'aria-expanded': isSelected,
          onClick: () => onSelect(isSelected ? null : row.run_id),
          onKeyDown: (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              onSelect(isSelected ? null : row.run_id)
            }
          },
          style: {
            display: 'grid',
            gridTemplateColumns: '92px minmax(120px, 1.1fr) minmax(150px, 1.5fr) 128px 74px 16px',
            alignItems: 'center',
            gap: 10,
            padding: '9px 12px 9px 10px',
            borderRadius: 8,
            cursor: 'pointer',
            fontSize: 12.5,
            background: isSelected ? 'var(--color-bg-tertiary, rgba(127,127,127,.14))' : 'transparent',
            borderLeft: `3px solid ${running ? COLORS.running : 'transparent'}`,
            outline: 'none',
          },
          onMouseEnter: (event) => {
            if (!isSelected) event.currentTarget.style.background = SOFT_BG
          },
          onMouseLeave: (event) => {
            if (!isSelected) event.currentTarget.style.background = 'transparent'
          },
        },
        statusPill(row.display),
        h(
          'div',
          { style: { minWidth: 0 } },
          h(
            'div',
            { style: { fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } },
            row.sample || row.run_id,
          ),
          h('div', { style: { fontSize: 10.5, color: MUTED, fontFamily: MONO, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, row.run_id),
        ),
        h(
          'div',
          { style: { minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 } },
          h(
            'div',
            { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } },
            row.step || (row.display === 'queued' ? '等待 worker' : '—'),
          ),
          stepDots(row),
        ),
        progressBar(row),
        h(
          'div',
          { style: { fontSize: 11, color: MUTED, textAlign: 'right', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' } },
          duration || '—',
          eta ? h('div', { style: { fontSize: 10 } }, eta) : null,
        ),
        h('span', { style: { color: MUTED, fontSize: 10, transform: isSelected ? 'rotate(90deg)' : 'none', transition: 'transform .15s ease' } }, '▶'),
      )
    }

    // -------------------------------------------------------------- detail
    function fact(label, value, mono) {
      return h(
        'div',
        { style: { display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 } },
        h('span', { style: { fontSize: 10.5, color: MUTED } }, label),
        h(
          'span',
          { style: { fontSize: 12, fontFamily: mono ? MONO : 'inherit', overflow: 'hidden', textOverflow: 'ellipsis' } },
          value === null || value === undefined || value === '' ? '—' : String(value),
        ),
      )
    }

    function detailPanel(detail) {
      if (!detail) {
        return h('div', { style: { padding: '10px 14px', fontSize: 12, color: MUTED } }, '读取任务详情…')
      }
      const record = detail.record || {}
      const progress = detail.progress || {}
      const error = record.error
      const steps = Array.isArray(progress.steps) ? progress.steps : []
      const verification = detail.verification
      const logTail = detail.log_tail

      return h(
        'div',
        {
          style: {
            margin: '2px 0 8px 10px',
            padding: '12px 14px',
            borderRadius: 8,
            border: BORDER,
            background: SOFT_BG,
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
          },
        },
        h(
          'div',
          {
            style: {
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '8px 18px',
            },
          },
          fact('run_id', record.run_id, true),
          fact('项目', record.project),
          fact('config', record.config),
          fact('状态', LABELS[record.display] || record.display),
          fact('开始', fmtClock(record.started_at)),
          fact('耗时', fmtDuration(record.duration_s)),
          fact('心跳', record.heartbeat_age_s === null ? '—' : fmtAgo(record.heartbeat_age_s)),
          fact('步数', record.step_index === null ? '—' : `${record.step_index} / ${record.step_total}`),
        ),
        error
          ? h(
              'div',
              {
                style: {
                  border: '1px solid color-mix(in srgb, var(--color-red-500, #ef4444) 35%, transparent)',
                  background: 'color-mix(in srgb, var(--color-red-500, #ef4444) 8%, transparent)',
                  borderRadius: 6,
                  padding: '8px 10px',
                  fontSize: 12,
                },
              },
              h('div', { style: { color: 'var(--color-red-500, #ef4444)', fontWeight: 600 } }, `[${error.code || 'ERROR'}] ${error.message || ''}`),
              error.step_id !== null && error.step_id !== undefined ? h('div', { style: { color: MUTED, marginTop: 2 } }, `出错步骤：${error.step_id}`) : null,
              error.retryable ? h('div', { style: { color: MUTED, marginTop: 2 } }, '可重试') : null,
              error.suggestion ? h('div', { style: { marginTop: 4 } }, `建议：${error.suggestion}`) : null,
            )
          : null,
        steps.length > 0
          ? h(
              'div',
              { style: { display: 'flex', flexWrap: 'wrap', gap: 6 } },
              steps.map((step, index) => {
                const active = record.display === 'running' && Number(progress.step_index) === index + 1
                const failed = record.display === 'failed' && Number(progress.step_index) === index + 1
                const color = failed ? COLORS.failed : step.skipped ? MUTED : active ? COLORS.running : COLORS.succeeded
                return h(
                  'span',
                  {
                    key: `${index}-${step.name}`,
                    title: step.peak_mem_mb ? `峰值内存 ${step.peak_mem_mb} MB` : '',
                    style: {
                      fontSize: 11,
                      padding: '2px 8px',
                      borderRadius: 999,
                      border: `1px solid color-mix(in srgb, ${color} 35%, transparent)`,
                      color,
                      whiteSpace: 'nowrap',
                    },
                  },
                  `${index + 1}. ${step.name}${step.skipped ? '（跳过）' : step.seconds ? ` · ${fmtDuration(step.seconds)}` : ''}`,
                )
              }),
            )
          : null,
        verification
          ? h(
              'div',
              { style: { fontSize: 12, color: verification.ok ? COLORS.succeeded : COLORS.failed } },
              `产物校验：${verification.ok ? '通过' : '未通过'}`,
            )
          : null,
        logTail
          ? h(
              'div',
              { style: { display: 'flex', flexDirection: 'column', gap: 4 } },
              h('div', { style: { fontSize: 10.5, color: MUTED } }, `日志尾部 · ${detail.log_path || ''}`),
              h(
                'pre',
                {
                  style: {
                    margin: 0,
                    padding: '8px 10px',
                    borderRadius: 6,
                    maxHeight: 220,
                    overflow: 'auto',
                    fontSize: 11,
                    lineHeight: 1.5,
                    fontFamily: MONO,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    background: 'var(--color-bg-primary, rgba(0,0,0,.15))',
                    border: BORDER,
                  },
                },
                logTail,
              ),
            )
          : h('div', { style: { fontSize: 11.5, color: MUTED } }, '暂无日志'),
      )
    }

    // -------------------------------------------------------------- groups
    function groupHeader(group, open, onToggle) {
      const counts = []
      if (group.running) counts.push(`${group.running} 运行`)
      if (group.queued) counts.push(`${group.queued} 排队`)
      if (group.failed) counts.push(`${group.failed} 失败`)
      if (!counts.length) counts.push(`${group.total} 个样本`)
      return h(
        'div',
        {
          role: 'button',
          tabIndex: 0,
          onClick: onToggle,
          onKeyDown: (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              onToggle()
            }
          },
          style: {
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '7px 10px',
            cursor: 'pointer',
            borderRadius: 6,
            userSelect: 'none',
          },
        },
        h('span', { style: { color: MUTED, fontSize: 9, transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .15s ease' } }, '▶'),
        h('span', { style: { fontWeight: 600, fontSize: 12.5 } }, group.key),
        group.state !== 'active'
          ? h(
              'span',
              {
                style: {
                  fontSize: 10.5,
                  padding: '1px 6px',
                  borderRadius: 4,
                  color: group.state === 'delivered' ? COLORS.succeeded : COLORS.cancelled,
                  border: `1px solid color-mix(in srgb, ${group.state === 'delivered' ? COLORS.succeeded : COLORS.cancelled} 30%, transparent)`,
                },
              },
              group.state === 'paused' ? '暂不排期' : '已交付',
            )
          : null,
        h('span', { style: { fontSize: 11, color: MUTED } }, counts.join(' · ')),
      )
    }

    // --------------------------------------------------------------- panel
    function Panel() {
      const [data, setData] = React.useState(null)
      const [error, setError] = React.useState(null)
      const [selected, setSelected] = React.useState(null)
      const [detail, setDetail] = React.useState(null)
      const [showFinished, setShowFinished] = React.useState(false)
      const [updatedAt, setUpdatedAt] = React.useState(null)
      const [closedGroups, setClosedGroups] = React.useState({})

      const poll = React.useCallback(async () => {
        if (document.hidden) return
        try {
          const response = await fetch(`/yifu-pm/jobs?limit=80&finished=${showFinished ? 1 : 0}`)
          if (!response.ok) throw new Error(`HTTP ${response.status}`)
          setData(await response.json())
          setUpdatedAt(new Date())
          setError(null)
        } catch (err) {
          setError(String((err && err.message) || err))
        }
      }, [showFinished])

      React.useEffect(() => {
        poll()
        const timer = setInterval(poll, POLL_MS)
        const onVisible = () => {
          if (!document.hidden) poll()
        }
        document.addEventListener('visibilitychange', onVisible)
        return () => {
          clearInterval(timer)
          document.removeEventListener('visibilitychange', onVisible)
        }
      }, [poll])

      React.useEffect(() => {
        if (!selected) {
          setDetail(null)
          return undefined
        }
        let alive = true
        const load = async () => {
          try {
            const response = await fetch(`/yifu-pm/job?run_id=${encodeURIComponent(selected)}&tail=60`)
            const payload = await response.json()
            if (alive) setDetail(payload)
          } catch {
            /* keep the previous detail */
          }
        }
        load()
        const timer = setInterval(load, POLL_MS)
        return () => {
          alive = false
          clearInterval(timer)
        }
      }, [selected])

      const jobs = (data && data.jobs) || []
      const worker = (data && data.worker) || {}
      const counts = (data && data.counts) || {}
      const legacyUrl = data && data.legacy_pm_url

      const groups = []
      const byProject = new Map()
      for (const job of jobs) {
        const key = job.project || '(未归项目)'
        if (!byProject.has(key)) byProject.set(key, [])
        byProject.get(key).push(job)
      }
      for (const [key, rows] of byProject) {
        const first = rows[0]
        groups.push({
          key,
          rows,
          state: first.project_state || 'active',
          total: rows.length,
          running: rows.filter((row) => row.display === 'running').length,
          queued: rows.filter((row) => row.display === 'queued').length,
          failed: rows.filter((row) => row.display === 'failed' || row.display === 'stalled').length,
        })
      }

      const isOpen = (group) => {
        if (Object.prototype.hasOwnProperty.call(closedGroups, group.key)) return !closedGroups[group.key]
        return group.state === 'active'
      }

      const headerCounts = [
        `运行 ${counts.running || 0}`,
        `排队 ${counts.queued || 0}`,
        `失败 ${(counts.failed || 0) + (counts.stalled || 0)}`,
      ]

      return h(
        'div',
        {
          style: {
            height: '100%',
            overflow: 'auto',
            padding: '16px 22px 28px',
            fontSize: 13,
            color: 'var(--color-fg-primary, inherit)',
          },
        },
        // sticky header
        h(
          'div',
          {
            style: {
              position: 'sticky',
              top: -16,
              zIndex: 2,
              margin: '-16px -22px 10px',
              padding: '16px 22px 10px',
              background: 'var(--color-bg-primary, Canvas)',
              borderBottom: BORDER,
            },
          },
          h(
            'div',
            { style: { display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' } },
            h('div', { style: { fontSize: 16, fontWeight: 650 } }, 'Yifu 流程队列'),
            h(
              'span',
              {
                style: {
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 11.5,
                  padding: '2px 8px',
                  borderRadius: 999,
                  color: worker.alive ? COLORS.succeeded : COLORS.stalled,
                  border: `1px solid color-mix(in srgb, ${worker.alive ? COLORS.succeeded : COLORS.stalled} 30%, transparent)`,
                },
              },
              h('span', {
                style: { width: 6, height: 6, borderRadius: '50%', background: worker.alive ? COLORS.succeeded : COLORS.stalled },
              }),
              worker.alive ? `worker 在线${worker.current_job_id ? ` · ${worker.current_job_id}` : ''}` : 'worker 离线',
            ),
            h('span', { style: { fontSize: 11.5, color: MUTED } }, headerCounts.join(' · ')),
            h('div', { style: { flex: '1 1 auto' } }),
            h(
              'button',
              {
                type: 'button',
                onClick: () => setShowFinished((value) => !value),
                style: {
                  padding: '2px 9px',
                  fontSize: 11.5,
                  borderRadius: 999,
                  cursor: 'pointer',
                  border: BORDER,
                  background: showFinished ? 'var(--color-bg-tertiary, rgba(127,127,127,.16))' : 'transparent',
                  color: 'inherit',
                },
              },
              showFinished ? '含已结束 ✓' : '含已结束',
            ),
            h(
              'button',
              {
                type: 'button',
                onClick: poll,
                style: { padding: '2px 9px', fontSize: 11.5, borderRadius: 999, cursor: 'pointer', border: BORDER, background: 'transparent', color: 'inherit' },
              },
              '刷新',
            ),
            legacyUrl
              ? h(
                  'a',
                  {
                    href: legacyUrl,
                    target: '_blank',
                    rel: 'noreferrer',
                    style: { fontSize: 11.5, color: 'var(--color-blue-500, #4aa8ff)', textDecoration: 'none' },
                  },
                  '旧 PM ↗',
                )
              : null,
            h('span', { style: { fontSize: 11, color: MUTED } }, updatedAt ? `更新于 ${updatedAt.toLocaleTimeString()}` : ''),
          ),
          error
            ? h('div', { style: { marginTop: 8, fontSize: 12, color: 'var(--color-red-500, #ef4444)' } }, `队列读取失败：${error}（数据来自 /yifu-pm/jobs）`)
            : null,
        ),
        groups.length === 0
          ? h(
              'div',
              { style: { padding: '28px 4px', fontSize: 12.5, color: MUTED } },
              showFinished ? '队列为空' : '没有未结束的任务；点右上角“含已结束”查看历史',
            )
          : null,
        groups.map((group) => {
          const open = isOpen(group)
          return h(
            'section',
            {
              key: group.key,
              style: {
                marginBottom: 10,
                border: BORDER,
                borderRadius: 10,
                padding: '4px 6px 6px',
                opacity: group.state === 'active' ? 1 : 0.55,
                background: group.state === 'active' ? 'transparent' : SOFT_BG,
              },
            },
            groupHeader(group, open, () =>
              setClosedGroups((previous) => ({ ...previous, [group.key]: open })),
            ),
            open
              ? h(
                  'div',
                  { style: { display: 'flex', flexDirection: 'column', gap: 1 } },
                  group.rows.map((row) =>
                    h(
                      'div',
                      { key: row.run_id },
                      jobRow(row, selected, setSelected),
                      selected === row.run_id ? detailPanel(detail) : null,
                    ),
                  ),
                )
              : null,
          )
        }),
        h(
          'div',
          { style: { marginTop: 18, fontSize: 10.5, color: MUTED, lineHeight: 1.6 } },
          '只读监视：数据来自 harness list，刷新不会启动任务。提交/取消请用 yifu_submit / yifu_cancel。',
        ),
      )
    }

    function Icon(props) {
      const size = (props && props.size) || 20
      const active = Boolean(props && props.active)
      return h(
        'svg',
        { viewBox: '0 0 24 24', width: size, height: size, 'aria-hidden': true, style: { display: 'block' } },
        h('rect', { x: 3, y: 5, width: 11, height: 3, rx: 1.5, fill: 'currentColor', opacity: active ? 1 : 0.7 }),
        h('rect', { x: 3, y: 11, width: 15, height: 3, rx: 1.5, fill: 'currentColor', opacity: active ? 1 : 0.7 }),
        h('rect', { x: 3, y: 17, width: 8, height: 3, rx: 1.5, fill: 'currentColor', opacity: active ? 1 : 0.7 }),
        h('circle', { cx: 18.5, cy: 6.5, r: 2.5, fill: 'var(--color-green-500, #22c55e)' }),
      )
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        ctx.slots.inject('sidebar.panellist', () =>
          ctx.slots.register(
            { name: 'sidebar.panellist', id: 'yifu-pm', order: 60, label: 'Yifu 队列' },
            Icon,
          ),
        )
        ctx.slots.inject('main', () =>
          ctx.slots.register({ name: 'main', key: 'yifu-pm' }, Panel),
        )
      },
    }
  },
})
