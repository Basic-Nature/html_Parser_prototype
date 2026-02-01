from playwright.sync_api import sync_playwright
import time
import os

html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'prompt_smoke.html'))
file_url = 'file://' + html_path.replace('\\', '/')
print('Opening', file_url)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(file_url)
    # wait for prompt to render
    time.sleep(0.5)

    # Click Cancel and observe effects
    try:
        page.evaluate("document.getElementById('btnCancelPrompt').click()")
    except Exception as e:
        print('Cancel click error:', e)
    time.sleep(0.2)
    modalClosed = page.evaluate('window._modalClosedId')
    socketEmit = page.evaluate('window._socketEmit')
    optCount = page.evaluate("document.querySelectorAll('.prompt-option').length")
    modalShown = page.evaluate('Boolean(window._modalShown)')
    print('After Cancel: modalClosed=', modalClosed, 'socketEmit=', socketEmit)
    print('  optionButtons=', optCount, 'modalShown=', modalShown)

    # Now show prompt again
    page.evaluate("(function(){ if(typeof handlePromptLog==='function') handlePromptLog({ type: 'prompt', message: '[PROMPT] Choose', context: { options: ['One','Two','Three'] } }); })()")
    time.sleep(0.2)
    # Click the first option button (auto-submits) instead of the Submit button
    try:
        page.evaluate("(function(){ var b=document.querySelector('.prompt-option'); if(b) b.click(); })()")
    except Exception as e:
        print('Option click error:', e)
    time.sleep(0.2)
    modalClosed2 = page.evaluate('window._modalClosedId')
    socketEmit2 = page.evaluate('window._socketEmit')
    optCount2 = page.evaluate("document.querySelectorAll('.prompt-option').length")
    modalShown2 = page.evaluate('Boolean(window._modalShown)')
    print('After Submit: modalClosed=', modalClosed2, 'socketEmit=', socketEmit2)
    print('  optionButtons=', optCount2, 'modalShown=', modalShown2)

    browser.close()
print('Done')
