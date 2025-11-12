const API_URL = 'http://localhost:8000';
let currentModel = null;
let availableModels = {};
let authToken = null;
let currentUser = null;

// 初期化フラグ
let isInitialized = false;

// 初期化
async function init() {
    if (isInitialized) return;
    isInitialized = true;

    // ローカルストレージからトークンを取得
    authToken = localStorage.getItem('authToken');
    currentUser = localStorage.getItem('currentUser');

    // イベントリスナーを設定（一度だけ）
    updateTemperatureDisplay();
    updateScoreThresholdDisplay();
    setupRAGToggle();

    if (authToken) {
        // 認証済みの場合、メイン画面を表示
        showMainScreen();
        await loadAvailableModels();
        await loadDocuments();
    } else {
        // 未認証の場合、サインイン画面を表示
        showAuthScreen();
    }
}

// 認証画面を表示
function showAuthScreen() {
    document.getElementById('authScreen').style.display = 'flex';
    document.getElementById('mainScreen').style.display = 'none';
}

// メイン画面を表示
function showMainScreen() {
    document.getElementById('authScreen').style.display = 'none';
    document.getElementById('mainScreen').style.display = 'flex';
    if (currentUser) {
        document.getElementById('currentUsername').textContent = currentUser;
    }
}

// サインイン・サインアップ切り替え
function switchAuthMode(mode) {
    const signinForm = document.getElementById('signinForm');
    const signupForm = document.getElementById('signupForm');
    const signinTab = document.getElementById('signinTab');
    const signupTab = document.getElementById('signupTab');

    if (mode === 'signin') {
        signinForm.style.display = 'block';
        signupForm.style.display = 'none';
        signinTab.classList.add('active');
        signupTab.classList.remove('active');
    } else {
        signinForm.style.display = 'none';
        signupForm.style.display = 'block';
        signinTab.classList.remove('active');
        signupTab.classList.add('active');
    }
}

// サインイン
async function signin() {
    const username = document.getElementById('signinUsername').value.trim();
    const password = document.getElementById('signinPassword').value;

    if (!username || !password) {
        alert('ユーザー名とパスワードを入力してください');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/signin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (response.ok) {
            authToken = data.token;
            currentUser = data.username;
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('currentUser', currentUser);
            showMainScreen();
            await loadAvailableModels();
            await loadDocuments();
        } else {
            alert(`エラー: ${data.detail}`);
        }
    } catch (error) {
        alert(`エラー: ${error.message}`);
    }
}

// サインアップ
async function signup() {
    const username = document.getElementById('signupUsername').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;

    if (!username || !email || !password || !confirmPassword) {
        alert('全ての項目を入力してください');
        return;
    }

    if (password !== confirmPassword) {
        alert('パスワードが一致しません');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();

        if (response.ok) {
            authToken = data.token;
            currentUser = data.username;
            localStorage.setItem('authToken', authToken);
            localStorage.setItem('currentUser', currentUser);
            showMainScreen();
            await loadAvailableModels();
            await loadDocuments();
        } else {
            alert(`エラー: ${data.detail}`);
        }
    } catch (error) {
        alert(`エラー: ${error.message}`);
    }
}

// サインアウト
async function signout() {
    try {
        await fetch(`${API_URL}/signout`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        authToken = null;
        currentUser = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('currentUser');
        showAuthScreen();

        // フォームをリセット
        document.getElementById('signinUsername').value = '';
        document.getElementById('signinPassword').value = '';
        document.getElementById('signupUsername').value = '';
        document.getElementById('signupEmail').value = '';
        document.getElementById('signupPassword').value = '';
        document.getElementById('signupConfirmPassword').value = '';
    } catch (error) {
        console.error('サインアウトエラー:', error);
    }
}

// APIリクエストにAuthorizationヘッダーを追加するヘルパー関数
function getAuthHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`
    };
}

// RAGトグルのセットアップ
function setupRAGToggle() {
    const useRAGCheckbox = document.getElementById('useRAG');
    const ragKLabel = document.getElementById('ragKLabel');
    const scoreThresholdLabel = document.getElementById('scoreThresholdLabel');

    useRAGCheckbox.onchange = (e) => {
        const isChecked = e.target.checked;
        ragKLabel.style.display = isChecked ? 'flex' : 'none';
        scoreThresholdLabel.style.display = isChecked ? 'flex' : 'none';
    };
}

// 利用可能なモデルを取得
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();
        availableModels = data.models;
        currentModel = data.current_model;

        const select = document.getElementById('modelSelect');
        select.innerHTML = '';

        for (const [id, info] of Object.entries(availableModels)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${info.description}`;
            if (id === currentModel) {
                option.selected = true;
            }
            select.appendChild(option);
        }

        updateModelStatus();
    } catch (error) {
        addSystemMessage('エラー: サーバーに接続できません。FastAPIサーバーが起動しているか確認してください。');
    }
}

// モデルをロード
async function loadSelectedModel() {
    const modelId = document.getElementById('modelSelect').value;
    if (!modelId) return;

    const statusEl = document.getElementById('modelStatus');
    statusEl.textContent = 'ロード中...';

    try {
        const response = await fetch(`${API_URL}/model/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });

        const data = await response.json();

        if (response.ok) {
            currentModel = data.model_id;
            addSystemMessage(`モデル「${data.model_info.description}」をロードしました。`);
            updateModelStatus();
        } else {
            addSystemMessage(`エラー: ${data.detail}`);
            statusEl.textContent = '';
        }
    } catch (error) {
        addSystemMessage(`エラー: ${error.message}`);
        statusEl.textContent = '';
    }
}

// モデルステータスを更新
function updateModelStatus() {
    const statusEl = document.getElementById('modelStatus');
    if (currentModel) {
        statusEl.textContent = '✓ ロード済み';
        statusEl.style.color = '#28a745';
    } else {
        statusEl.textContent = '';
    }
}

// Temperature表示を更新
function updateTemperatureDisplay() {
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    tempSlider.oninput = (e) => {
        tempValue.textContent = e.target.value;
    };
}

// スコア閾値表示を更新
function updateScoreThresholdDisplay() {
    const scoreSlider = document.getElementById('scoreThreshold');
    const scoreValue = document.getElementById('scoreThresholdValue');
    scoreSlider.oninput = (e) => {
        const value = e.target.value;
        scoreValue.textContent = value === '0' ? 'なし' : value;
    };
}

// ファイルアップロード
async function uploadFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const loading = document.getElementById('loading');
    loading.classList.add('active');

    try {
        const response = await fetch(`${API_URL}/documents/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            addSystemMessage(`ファイル「${file.name}」をアップロードしました。`);
            await loadDocuments();
        } else {
            if (response.status === 401) {
                alert('認証が切れました。再度サインインしてください。');
                signout();
            } else {
                addSystemMessage(`エラー: ${data.detail}`);
            }
        }
    } catch (error) {
        addSystemMessage(`エラー: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        event.target.value = '';
    }
}

// ドキュメント一覧を取得
async function loadDocuments() {
    try {
        const response = await fetch(`${API_URL}/documents/list`);
        const data = await response.json();

        const listEl = document.getElementById('documentsList');

        if (data.count === 0) {
            listEl.innerHTML = '<p style="color: #999; text-align: center; font-size: 13px;">ドキュメントなし</p>';
        } else {
            listEl.innerHTML = '';
            data.documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'document-item';
                item.innerHTML = `
                    <div class="name">${doc.filename}</div>
                    <div class="info">チャンク: ${doc.chunks} | 文字数: ${doc.characters.toLocaleString()}</div>
                `;
                listEl.appendChild(item);
            });
        }
    } catch (error) {
        console.error('ドキュメント一覧の取得エラー:', error);
    }
}

// 全ドキュメント削除
async function deleteAllDocuments() {
    if (!confirm('全てのドキュメントを削除しますか？')) return;

    try {
        const response = await fetch(`${API_URL}/documents/delete`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        const data = await response.json();

        if (response.ok) {
            addSystemMessage('全てのドキュメントを削除しました。');
            await loadDocuments();
        } else {
            if (response.status === 401) {
                alert('認証が切れました。再度サインインしてください。');
                signout();
            } else {
                addSystemMessage(`エラー: ${data.detail}`);
            }
        }
    } catch (error) {
        addSystemMessage(`エラー: ${error.message}`);
    }
}

// メッセージを送信
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message) return;
    if (!currentModel) {
        addSystemMessage('先にモデルをロードしてください。');
        return;
    }

    // ユーザーメッセージを追加
    addMessage('user', message);
    input.value = '';

    // 送信ボタンを無効化
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;

    // ローディング表示
    const loading = document.getElementById('loading');
    loading.classList.add('active');

    try {
        const maxLength = parseInt(document.getElementById('maxLength').value);
        const temperature = parseFloat(document.getElementById('temperature').value);
        const useRAG = document.getElementById('useRAG').checked;
        const ragK = parseInt(document.getElementById('ragK').value);
        const scoreThreshold = parseFloat(document.getElementById('scoreThreshold').value);

        const requestBody = {
            message: message,
            model_id: currentModel,
            max_length: maxLength,
            temperature: temperature,
            use_rag: useRAG,
            rag_k: ragK
        };

        // スコア閾値が0の場合はnull（フィルタリングなし）、それ以外は値を設定
        if (useRAG) {
            requestBody.score_threshold = scoreThreshold === 0 ? null : scoreThreshold;
        }

        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (response.ok) {
            addMessage('bot', data.response, data.retrieved_docs);
        } else {
            if (response.status === 401) {
                alert('認証が切れました。再度サインインしてください。');
                signout();
            } else {
                addSystemMessage(`エラー: ${data.detail}`);
            }
        }
    } catch (error) {
        addSystemMessage(`エラー: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        sendButton.disabled = false;
        input.focus();
    }
}

// メッセージを追加
function addMessage(type, content, retrievedDocs = null) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);

    // 検索されたドキュメントを表示
    if (retrievedDocs && retrievedDocs.length > 0) {
        const docsDiv = document.createElement('div');
        docsDiv.className = 'retrieved-docs';

        docsDiv.innerHTML = '<div class="doc-title">📚 参考にした情報:</div>';

        retrievedDocs.forEach((doc, index) => {
            const docContent = document.createElement('div');
            docContent.className = 'doc-content';
            docContent.textContent = `[${index + 1}] ${doc.source}: ${doc.content.substring(0, 100)}...`;
            docsDiv.appendChild(docContent);
        });

        contentDiv.appendChild(docsDiv);
    }

    messagesContainer.appendChild(messageDiv);

    // スクロールを最下部に
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// システムメッセージを追加
function addSystemMessage(content) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Enter キーで送信
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// 初期化実行
init();
