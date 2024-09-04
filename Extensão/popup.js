document.getElementById('classify-btn').addEventListener('click', async () => {
    console.log('Botão clicado');
    
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab) {
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: getSelectedText,
        }, async (results) => {
            let selectedText = results[0]?.result;
            
            if (selectedText) {
                console.log('Texto Selecionado:', selectedText);
                document.getElementById('status').innerText = 'Classificando...';

                try {
                    let response = await fetch('http://localhost:5000/classify', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: selectedText })
                    });

                    let result = await response.json();
                    console.log('Resultado da API:', result);
                    document.getElementById('status').innerText = result.isFake ? 'Esta é provavelmente uma fake news. Pesquise em sites de confiança para ter certeza.' : 'Esta é provavelmente uma notícia verdadeira. Pesquise em sites de confiança para ter certeza.';
                } catch (error) {
                    console.error('Erro ao chamar a API:', error);
                    document.getElementById('status').innerText = 'Erro ao classificar a noticia.';
                }
            } else {
                document.getElementById('status').innerText = 'Nenhum texto selecionado.';
            }
        });
    } else {
        console.error('Nenhuma aba encontrada.');
    }
});

function getSelectedText() {
    return window.getSelection().toString();
}
