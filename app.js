// Gold Trading Analytics Terminal - Frontend Script Logic
// Handles table pagination, chart rendering, custom natural language queries, and XGBoost forecasts.

let currentPage = 1;
let limit = 15;
let searchQuery = "";
let chartInstance = null;
const API_URL = ""; // Relative path to local server

// Initialize dashboard on load
document.addEventListener("DOMContentLoaded", () => {
    fetchStats();
    fetchTableData();
    fetchChartData();
    runXGBoostForecast();
});

// Fetch stats cards
async function fetchStats() {
    try {
        const res = await fetch(`${API_URL}/api/stats`);
        const data = await res.json();
        
        document.getElementById('stat-total').innerText = Number(data.total_records).toLocaleString();
        document.getElementById('stat-range').innerText = `Khoảng thời gian: ${data.min_date} đến ${data.max_date}`;
        document.getElementById('stat-latest').innerText = `${Number(data.latest_close).toLocaleString()} USD`;
        document.getElementById('stat-latest-date').innerText = `Ngày: ${data.latest_date}`;
        document.getElementById('stat-max-high').innerText = `${Number(data.max_high).toLocaleString()} USD`;
        document.getElementById('stat-min-low').innerText = `${Number(data.min_low).toLocaleString()} USD`;
    } catch (err) {
        console.error("Error fetching stats:", err);
    }
}

// Fetch paginated table data
async function fetchTableData() {
    try {
        const res = await fetch(`${API_URL}/api/data?page=${currentPage}&limit=${limit}&search=${encodeURIComponent(searchQuery)}`);
        const result = await res.json();
        
        const tableBody = document.getElementById('tableBody');
        tableBody.innerHTML = "";

        if (result.data.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">Không tìm thấy dữ liệu phù hợp</td></tr>`;
            document.getElementById('recordRange').innerText = "Hiển thị 0 bản ghi";
            document.getElementById('pageNumber').innerText = "Trang 0 / 0";
            document.getElementById('btnPrev').disabled = true;
            document.getElementById('btnNext').disabled = true;
            return;
        }

        const maxVol = Math.max(...result.data.map(d => d.Volume), 1);

        result.data.forEach(row => {
            const tr = document.createElement('tr');
            const volPercent = Math.min((row.Volume / maxVol) * 100, 100);
            
            tr.innerHTML = `
                <td style="font-weight: 600; color: #fff;">${row.Date}</td>
                <td>${Number(row.Open).toFixed(2)}</td>
                <td style="color: var(--green); font-weight: 500;">${Number(row.High).toFixed(2)}</td>
                <td style="color: var(--red); font-weight: 500;">${Number(row.Low).toFixed(2)}</td>
                <td style="font-weight: 700; color: var(--accent-gold);">${Number(row.Close).toFixed(2)}</td>
                <td>
                    <div class="vol-bar-container">
                        <div class="vol-bar" style="width: ${volPercent}%"></div>
                    </div>
                    <span style="font-size: 0.8rem; font-family: var(--font-mono);">${Number(row.Volume).toLocaleString()}</span>
                </td>
                <td>${Number(row.Open_interest).toLocaleString()}</td>
            `;
            tableBody.appendChild(tr);
        });

        const fromRecord = (currentPage - 1) * limit + 1;
        const toRecord = Math.min(currentPage * limit, result.total);
        document.getElementById('recordRange').innerText = `Hiển thị ${fromRecord}-${toRecord} của ${result.total.toLocaleString()} bản ghi`;
        document.getElementById('pageNumber').innerText = `Trang ${currentPage} / ${result.total_pages}`;
        
        document.getElementById('btnPrev').disabled = currentPage === 1;
        document.getElementById('btnNext').disabled = currentPage === result.total_pages;
    } catch (err) {
        console.error("Error fetching table data:", err);
    }
}

// Change page
function changePage(direction) {
    currentPage += direction;
    fetchTableData();
}

// Handle live search
let searchTimeout;
function handleSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        searchQuery = document.getElementById('searchInput').value;
        currentPage = 1;
        fetchTableData();
    }, 300);
}

// Fetch chart data and render Chart.js
async function fetchChartData() {
    try {
        const res = await fetch(`${API_URL}/api/chart`);
        const data = await res.json();
        
        const ctx = document.getElementById('goldChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(255, 199, 44, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 199, 44, 0.0)');

        chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.dates,
                        datasets: [
                            {
                                label: 'Giá đóng cửa Close (USD)',
                                data: data.prices,
                                borderColor: '#ffc72c',
                                borderWidth: 2,
                                pointRadius: 0,
                                pointHoverRadius: 6,
                                fill: true,
                                backgroundColor: gradient,
                                tension: 0.1,
                                yAxisID: 'y'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index',
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                backgroundColor: '#111827',
                                titleFont: { family: 'Plus Jakarta Sans', weight: 'bold' },
                                bodyFont: { family: 'Plus Jakarta Sans' },
                                borderColor: 'rgba(255, 255, 255, 0.1)',
                                borderWidth: 1,
                                displayColors: false,
                                padding: 12
                            }
                        },
                        scales: {
                            x: {
                                grid: {
                                    display: false
                                },
                                ticks: {
                                    color: '#9ca3af',
                                    font: { family: 'Plus Jakarta Sans', size: 10 },
                                    maxTicksLimit: 8
                                }
                            },
                            y: {
                                grid: {
                                    color: 'rgba(255, 255, 255, 0.05)'
                                },
                                ticks: {
                                    color: '#9ca3af',
                                    font: { family: 'Plus Jakarta Sans', size: 10 }
                                }
                            }
                        }
                    }
                });
    } catch (err) {
        console.error("Error creating chart:", err);
    }
}

// Run AI Natural Language Query
async function runNLPQuery() {
    const prompt = document.getElementById('nlpQuery').value.trim();
    const resultsDiv = document.getElementById('sqlResults');
    
    if (!prompt) {
        resultsDiv.innerHTML = `<div class="sql-result-error">Vui lòng nhập câu hỏi phân tích bằng ngôn ngữ tự nhiên.</div>`;
        return;
    }

    resultsDiv.innerHTML = `<div style="padding: 1.5rem; text-align: center; color: var(--accent-gold);">
        <span style="font-family: var(--font-mono); font-size: 0.9rem;" class="pulse">AI đang phân tích câu hỏi & truy vấn...</span>
    </div>`;

    try {
        const res = await fetch(`${API_URL}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: prompt })
        });
        
        const data = await res.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="sql-result-error">Không thể phân tích yêu cầu này. Chi tiết: ${data.error}</div>`;
            return;
        }

        if (!data.success || data.count === 0) {
            resultsDiv.innerHTML = `
                <div class="sql-result-success">Phân tích hoàn tất: Không tìm thấy kết quả phù hợp.</div>
            `;
            return;
        }

        let html = `
            <div class="sql-result-success">Kết quả phân tích thành công: Tìm thấy ${data.count} bản ghi phù hợp.</div>
            <div style="overflow-x: auto; width: 100%;">
                <table style="border-radius: 0; font-family: var(--font-mono); font-size: 0.75rem;">
                    <thead>
                        <tr style="background: rgba(255,255,255,0.02)">
                            ${data.columns.map(col => `<th style="padding: 0.5rem 0.75rem;">${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.rows.forEach(row => {
            html += `<tr style="border-bottom: 1px solid rgba(255,255,255,0.03)">`;
            data.columns.forEach(col => {
                let val = row[col];
                if (col.toLowerCase() === 'volume') {
                    // Force international comma separators and add Contract suffix to prevent dot/decimal confusion
                    val = `<span style="font-weight: 600; color: #fff;">${Number(val).toLocaleString('en-US')}</span> <span style="font-size: 0.7rem; color: var(--text-secondary);">HĐ</span>`;
                } else if (typeof val === 'number' && !Number.isInteger(val)) {
                    val = val.toFixed(3);
                } else if (typeof val === 'number') {
                    val = val.toLocaleString();
                }
                html += `<td style="padding: 0.5rem 0.75rem;">${val}</td>`;
            });
            html += `</tr>`;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;
        resultsDiv.innerHTML = html;
    } catch (err) {
        resultsDiv.innerHTML = `<div class="sql-result-error">Lỗi kết nối tới Server: ${err.message}</div>`;
    }
}

// Set Quick NL Query Templates
function setNLQueryTemplate(id) {
    const queryInput = document.getElementById('nlpQuery');
    if (id === 1) {
        queryInput.value = "Top 10 ngày có khối lượng giao dịch nhiều nhất";
    } else if (id === 2) {
        queryInput.value = "Hiển thị 15 ngày có RSI lớn hơn 70 và độ biến động nhỏ hơn 1.0";
    } else if (id === 3) {
        queryInput.value = "Cho biết giá đóng cửa trung bình và rsi trung bình trong năm 2025";
    } else if (id === 4) {
        queryInput.value = "Liệt kê 10 ngày có độ biến động lớn nhất";
    }
    runNLPQuery();
}

function clearNLP() {
    document.getElementById('nlpQuery').value = "";
    document.getElementById('sqlResults').innerHTML = `<div class="sql-result-empty">Nhập câu hỏi và nhấn "Phân tích" để hiển thị dữ liệu...</div>`;
}

// --- XGBOOST DYNAMIC FORECAST MODULE ---
async function runXGBoostForecast() {
    const predictDate = document.getElementById('predictDate').value;
    const resultCard = document.getElementById('predictResultCard');
    
    if (!predictDate) {
        alert("Vui lòng chọn ngày để dự báo!");
        return;
    }

    resultCard.style.display = "flex";
    resultCard.innerHTML = `<div style="text-align: center; color: var(--accent-gold); padding: 1rem; font-family: var(--font-mono); font-size: 0.85rem;">
        <span class="pulse">Running XGBoost Inference Engine...</span>
    </div>`;

    try {
        const res = await fetch(`${API_URL}/api/predict?date=${predictDate}`);
        const data = await res.json();

        if (data.error) {
            resultCard.innerHTML = `<div style="color: var(--red); font-size: 0.85rem; font-weight: 600;">
                Lỗi: ${data.error}
            </div>`;
            return;
        }

        // Compile result display
        const isUp = (data.prediction === "TĂNG");
        const trendClass = isUp ? "up" : "down";
        const trendText = isUp ? "📈 TĂNG (UP)" : "📉 GIẢM (DOWN)";
        
        // Determine evaluation badge
        let validationBadge = "";
        let validationSection = "";
        
        if (data.actual_close !== null) {
            const correctClass = data.is_correct ? "up" : "down";
            const correctText = data.is_correct ? "✓ Dự báo Đúng" : "✗ Dự báo Chưa Đúng";
            validationBadge = `<span class="predict-badge ${correctClass}">${correctText}</span>`;
            
            validationSection = `
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 0.75rem; margin-top: 0.25rem;">
                    <span style="color: var(--text-secondary)">Giá thực tế ngày đó:</span>
                    <span style="font-weight: 700; color: #fff;">${data.actual_close.toFixed(2)} USD (${data.actual_outcome})</span>
                </div>
            `;
        } else {
            validationBadge = `<span class="predict-badge future">🔮 Dự báo Tương lai</span>`;
        }

        resultCard.innerHTML = `
            <div class="predict-header">
                <span style="font-size: 0.75rem; color: var(--text-secondary); font-weight: 700; text-transform: uppercase;">Dự báo cho ngày: ${data.target_date}</span>
                ${validationBadge}
            </div>

            <div class="predict-trend ${trendClass}">
                ${trendText}
            </div>

            <div class="prob-bar-wrapper">
                <div class="prob-labels">
                    <span style="color: var(--green)">Tăng: ${data.prob_up}%</span>
                    <span style="color: var(--red)">Giảm: ${data.prob_down}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-up" style="width: ${data.prob_up}%"></div>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 0.5rem;">
                <span style="color: var(--text-secondary)">Giá hôm trước (${data.prev_date}):</span>
                <span style="font-weight: 700; color: var(--accent-gold);">${data.prev_close.toFixed(2)} USD</span>
            </div>

            ${validationSection}

            <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-secondary); margin-top: 0.25rem;">Chỉ số đầu vào từ ngày ${data.prev_date}:</div>
            <div class="indicator-grid">
                <div class="indicator-box">
                    <span class="ind-label">MA10</span>
                    <span class="ind-value">${data.indicators.MA10.toFixed(1)}</span>
                </div>
                <div class="indicator-box">
                    <span class="ind-label">MA30</span>
                    <span class="ind-value">${data.indicators.MA30.toFixed(1)}</span>
                </div>
                <div class="indicator-box">
                    <span class="ind-label">MA50</span>
                    <span class="ind-value">${data.indicators.MA50.toFixed(1)}</span>
                </div>
                <div class="indicator-box">
                    <span class="ind-label">RSI14</span>
                    <span class="ind-value">${data.indicators.RSI14.toFixed(1)}%</span>
                </div>
                <div class="indicator-box">
                    <span class="ind-label">MACD</span>
                    <span class="ind-value">${data.indicators.MACD.toFixed(3)}</span>
                </div>
                <div class="indicator-box">
                    <span class="ind-label">Volatility</span>
                    <span class="ind-value">${data.indicators.Volatility.toFixed(3)}%</span>
                </div>
            </div>
        `;

    } catch (err) {
        resultCard.innerHTML = `<div style="color: var(--red); font-size: 0.85rem; font-weight: 600;">
            Lỗi kết nối Server: ${err.message}
        </div>`;
    }
}
