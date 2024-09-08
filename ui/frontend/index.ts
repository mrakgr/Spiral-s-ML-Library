import '@shoelace-style/shoelace/dist/themes/light.css';
import '@shoelace-style/shoelace/dist/components/button/button.js';
import '@shoelace-style/shoelace/dist/components/input/input.js';
import '@shoelace-style/shoelace/dist/components/tab/tab.js';
import '@shoelace-style/shoelace/dist/components/tab-group/tab-group.js';
import '@shoelace-style/shoelace/dist/components/tab-panel/tab-panel.js';
import '@shoelace-style/shoelace/dist/components/icon/icon.js';
import '@shoelace-style/shoelace/dist/components/select/select.js';
import '@shoelace-style/shoelace/dist/components/option/option.js';
import '@shoelace-style/shoelace/dist/components/range/range.js';
import { setBasePath } from '@shoelace-style/shoelace/dist/utilities/base-path.js';

setBasePath('/bundles/shoelace/');

import "./rps"; // Needed to import the rps web components.
import "./leduc";
import "./leduc_train";
import "./leduc_full";
import "./hu_nl_holdem";
import { LitElement, PropertyValueMap, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { gap } from './utils'

@customElement('games-ui')
class Games_UI extends LitElement {
    static styles = css`
        :host {
            display: block;
            height: 100vh;
            box-sizing: border-box;
        }

        sl-tab-group,
        sl-tab-group::part(base),
        sl-tab-group::part(body),
        sl-tab-panel,
        sl-tab-panel::part(base),
        sl-tab-panel::part(body) {
            height: 100%;
        }

        span.tab {
            font-size: var(--sl-font-size-x-large);
        }

        /* sl-tab-panel::part(base) {
            padding: 10px;
        }

        sl-tab::part(base)
        {
            padding: 15px;
        } */

        .train-area {
            height: 100%;
            padding: 5px;
            border: solid 5px black;
        }
    `

    render() {
        return html`
            <sl-tab-group>
                <sl-tab slot="nav" panel="rps"><span class="tab">RPS</span></sl-tab>
                <sl-tab slot="nav" panel="leduc"><span class="tab">Leduc</span></sl-tab>
                <sl-tab slot="nav" panel="leduc-train"><span class="tab">Leduc (Train)</span></sl-tab>
                <sl-tab slot="nav" panel="leduc-full" active><span class="tab">Leduc (Full)</span></sl-tab>
                <sl-tab slot="nav" panel="nl-holdem"><span class="tab">NL Holdem</span></sl-tab>
                <sl-tab-panel name="rps">
                    <rps-ui></rps-ui>
                </sl-tab-panel>
                <sl-tab-panel name="leduc">
                    <leduc-ui></leduc-ui>
                </sl-tab-panel>
                <sl-tab-panel name="leduc-train">
                    <leduc-train-ui></leduc-train-ui>
                </sl-tab-panel>
                <sl-tab-panel name="leduc-full">
                    <leduc-full-ui></leduc-full-ui>
                </sl-tab-panel>
                <sl-tab-panel name="nl-holdem">
                    <nl-holdem-ui></nl-holdem-ui>
                </sl-tab-panel>
            </sl-tab-group>
        `
    }
}