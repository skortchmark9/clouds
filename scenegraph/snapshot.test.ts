import { MatchImageSnapshotOptions } from 'jest-image-snapshot';
import puppeteer, { Browser, Page } from 'puppeteer';


jest.setTimeout(1e9);

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

        const startLong = -99.927038;
        const endLong = -96.727038
        const startLat = 37.794000;
        const endLat = 39.194000;
        let total = 0;

        for (let i = startLat; i < endLat; i += 0.1) {
            for (let j = startLong; j < endLong; j += 0.1) {
                total += 1;
                let pitch = 0;
                await page.evaluate((i, j, pitch) => {
                    // ignore ts error on this lines
                    // @ts-ignore
                    window.__goto({ latitude: i, longitude: j, pitch })
                }, i, j, pitch);
                await page.screenshot({ path: `screenshots/screenshot_${i.toFixed(5)}_${j.toFixed(5)}@${pitch}.png`, fullPage: true});

                pitch = 90;
                await page.evaluate((i, j, pitch) => {
                    // ignore ts error on this lines
                    // @ts-ignore
                    window.__goto({ latitude: i, longitude: j, pitch })
                }, i, j, pitch);
                await page.screenshot({ path: `screenshots/screenshot_${i.toFixed(5)}_${j.toFixed(5)}@${pitch}.png`, fullPage: true});

                if (total % 10 === 0) {
                    console.log(`Made ${total} images...`);
                }
            }
        }
        console.log(total);
    });
});
