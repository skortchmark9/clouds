import { MatchImageSnapshotOptions } from 'jest-image-snapshot';
import puppeteer, { Browser, Page } from 'puppeteer';


jest.setTimeout(10000);

function delay(time) {
    return new Promise(function(resolve) { 
        setTimeout(resolve, time)
    });
 }

describe('Google', () => {
    let page: Page;
    beforeAll(async () => {
        const browser: Browser = await puppeteer.launch({
            headless: true,
            defaultViewport: {
                width: 1440,
                height: 788,
            },
        });
        page = await browser.newPage();
        
        await page.goto('http://127.0.0.1:5173/');
    });
    
    it('should be titled.', async () => {
        await page.waitForNetworkIdle();
        await page.screenshot({ path: 'screenshots/screenshot.png', fullPage: true });
        await expect(page.title()).resolves.toMatch('deck.gl Example');
        await page.screenshot({ path: 'screenshots/screenshot.png', fullPage: true });
        await page.keyboard.press('p');
        await page.screenshot({ path: 'screenshots/screenshot1.png', fullPage: true });
    });
});
