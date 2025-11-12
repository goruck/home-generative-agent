function e(e,t,s,n){var r,i=arguments.length,o=i<3?t:null===n?n=Object.getOwnPropertyDescriptor(t,s):n;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)o=Reflect.decorate(e,t,s,n);else for(var l=e.length-1;l>=0;l--)(r=e[l])&&(o=(i<3?r(o):i>3?r(t,s,o):r(t,s))||o);return i>3&&o&&Object.defineProperty(t,s,o),o}"function"==typeof SuppressedError&&SuppressedError;
/**
 * @license
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const t=globalThis,s=t.ShadowRoot&&(void 0===t.ShadyCSS||t.ShadyCSS.nativeShadow)&&"adoptedStyleSheets"in Document.prototype&&"replace"in CSSStyleSheet.prototype,n=Symbol(),r=new WeakMap;let i=class{constructor(e,t,s){if(this._$cssResult$=!0,s!==n)throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");this.cssText=e,this.t=t}get styleSheet(){let e=this.o;const t=this.t;if(s&&void 0===e){const s=void 0!==t&&1===t.length;s&&(e=r.get(t)),void 0===e&&((this.o=e=new CSSStyleSheet).replaceSync(this.cssText),s&&r.set(t,e))}return e}toString(){return this.cssText}};const o=s?e=>e:e=>e instanceof CSSStyleSheet?(e=>{let t="";for(const s of e.cssRules)t+=s.cssText;return(e=>new i("string"==typeof e?e:e+"",void 0,n))(t)})(e):e,{is:l,defineProperty:a,getOwnPropertyDescriptor:c,getOwnPropertyNames:h,getOwnPropertySymbols:p,getPrototypeOf:u}=Object,d=globalThis,g=d.trustedTypes,f=g?g.emptyScript:"",k=d.reactiveElementPolyfillSupport,m=(e,t)=>e,x={toAttribute(e,t){switch(t){case Boolean:e=e?f:null;break;case Object:case Array:e=null==e?e:JSON.stringify(e)}return e},fromAttribute(e,t){let s=e;switch(t){case Boolean:s=null!==e;break;case Number:s=null===e?null:Number(e);break;case Object:case Array:try{s=JSON.parse(e)}catch(e){s=null}}return s}},b=(e,t)=>!l(e,t),w={attribute:!0,type:String,converter:x,reflect:!1,useDefault:!1,hasChanged:b};
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */Symbol.metadata??=Symbol("metadata"),d.litPropertyMetadata??=new WeakMap;let $=class extends HTMLElement{static addInitializer(e){this._$Ei(),(this.l??=[]).push(e)}static get observedAttributes(){return this.finalize(),this._$Eh&&[...this._$Eh.keys()]}static createProperty(e,t=w){if(t.state&&(t.attribute=!1),this._$Ei(),this.prototype.hasOwnProperty(e)&&((t=Object.create(t)).wrapped=!0),this.elementProperties.set(e,t),!t.noAccessor){const s=Symbol(),n=this.getPropertyDescriptor(e,s,t);void 0!==n&&a(this.prototype,e,n)}}static getPropertyDescriptor(e,t,s){const{get:n,set:r}=c(this.prototype,e)??{get(){return this[t]},set(e){this[t]=e}};return{get:n,set(t){const i=n?.call(this);r?.call(this,t),this.requestUpdate(e,i,s)},configurable:!0,enumerable:!0}}static getPropertyOptions(e){return this.elementProperties.get(e)??w}static _$Ei(){if(this.hasOwnProperty(m("elementProperties")))return;const e=u(this);e.finalize(),void 0!==e.l&&(this.l=[...e.l]),this.elementProperties=new Map(e.elementProperties)}static finalize(){if(this.hasOwnProperty(m("finalized")))return;if(this.finalized=!0,this._$Ei(),this.hasOwnProperty(m("properties"))){const e=this.properties,t=[...h(e),...p(e)];for(const s of t)this.createProperty(s,e[s])}const e=this[Symbol.metadata];if(null!==e){const t=litPropertyMetadata.get(e);if(void 0!==t)for(const[e,s]of t)this.elementProperties.set(e,s)}this._$Eh=new Map;for(const[e,t]of this.elementProperties){const s=this._$Eu(e,t);void 0!==s&&this._$Eh.set(s,e)}this.elementStyles=this.finalizeStyles(this.styles)}static finalizeStyles(e){const t=[];if(Array.isArray(e)){const s=new Set(e.flat(1/0).reverse());for(const e of s)t.unshift(o(e))}else void 0!==e&&t.push(o(e));return t}static _$Eu(e,t){const s=t.attribute;return!1===s?void 0:"string"==typeof s?s:"string"==typeof e?e.toLowerCase():void 0}constructor(){super(),this._$Ep=void 0,this.isUpdatePending=!1,this.hasUpdated=!1,this._$Em=null,this._$Ev()}_$Ev(){this._$ES=new Promise(e=>this.enableUpdating=e),this._$AL=new Map,this._$E_(),this.requestUpdate(),this.constructor.l?.forEach(e=>e(this))}addController(e){(this._$EO??=new Set).add(e),void 0!==this.renderRoot&&this.isConnected&&e.hostConnected?.()}removeController(e){this._$EO?.delete(e)}_$E_(){const e=new Map,t=this.constructor.elementProperties;for(const s of t.keys())this.hasOwnProperty(s)&&(e.set(s,this[s]),delete this[s]);e.size>0&&(this._$Ep=e)}createRenderRoot(){const e=this.shadowRoot??this.attachShadow(this.constructor.shadowRootOptions);return((e,n)=>{if(s)e.adoptedStyleSheets=n.map(e=>e instanceof CSSStyleSheet?e:e.styleSheet);else for(const s of n){const n=document.createElement("style"),r=t.litNonce;void 0!==r&&n.setAttribute("nonce",r),n.textContent=s.cssText,e.appendChild(n)}})(e,this.constructor.elementStyles),e}connectedCallback(){this.renderRoot??=this.createRenderRoot(),this.enableUpdating(!0),this._$EO?.forEach(e=>e.hostConnected?.())}enableUpdating(e){}disconnectedCallback(){this._$EO?.forEach(e=>e.hostDisconnected?.())}attributeChangedCallback(e,t,s){this._$AK(e,s)}_$ET(e,t){const s=this.constructor.elementProperties.get(e),n=this.constructor._$Eu(e,s);if(void 0!==n&&!0===s.reflect){const r=(void 0!==s.converter?.toAttribute?s.converter:x).toAttribute(t,s.type);this._$Em=e,null==r?this.removeAttribute(n):this.setAttribute(n,r),this._$Em=null}}_$AK(e,t){const s=this.constructor,n=s._$Eh.get(e);if(void 0!==n&&this._$Em!==n){const e=s.getPropertyOptions(n),r="function"==typeof e.converter?{fromAttribute:e.converter}:void 0!==e.converter?.fromAttribute?e.converter:x;this._$Em=n;const i=r.fromAttribute(t,e.type);this[n]=i??this._$Ej?.get(n)??i,this._$Em=null}}requestUpdate(e,t,s){if(void 0!==e){const n=this.constructor,r=this[e];if(s??=n.getPropertyOptions(e),!((s.hasChanged??b)(r,t)||s.useDefault&&s.reflect&&r===this._$Ej?.get(e)&&!this.hasAttribute(n._$Eu(e,s))))return;this.C(e,t,s)}!1===this.isUpdatePending&&(this._$ES=this._$EP())}C(e,t,{useDefault:s,reflect:n,wrapped:r},i){s&&!(this._$Ej??=new Map).has(e)&&(this._$Ej.set(e,i??t??this[e]),!0!==r||void 0!==i)||(this._$AL.has(e)||(this.hasUpdated||s||(t=void 0),this._$AL.set(e,t)),!0===n&&this._$Em!==e&&(this._$Eq??=new Set).add(e))}async _$EP(){this.isUpdatePending=!0;try{await this._$ES}catch(e){Promise.reject(e)}const e=this.scheduleUpdate();return null!=e&&await e,!this.isUpdatePending}scheduleUpdate(){return this.performUpdate()}performUpdate(){if(!this.isUpdatePending)return;if(!this.hasUpdated){if(this.renderRoot??=this.createRenderRoot(),this._$Ep){for(const[e,t]of this._$Ep)this[e]=t;this._$Ep=void 0}const e=this.constructor.elementProperties;if(e.size>0)for(const[t,s]of e){const{wrapped:e}=s,n=this[t];!0!==e||this._$AL.has(t)||void 0===n||this.C(t,void 0,s,n)}}let e=!1;const t=this._$AL;try{e=this.shouldUpdate(t),e?(this.willUpdate(t),this._$EO?.forEach(e=>e.hostUpdate?.()),this.update(t)):this._$EM()}catch(t){throw e=!1,this._$EM(),t}e&&this._$AE(t)}willUpdate(e){}_$AE(e){this._$EO?.forEach(e=>e.hostUpdated?.()),this.hasUpdated||(this.hasUpdated=!0,this.firstUpdated(e)),this.updated(e)}_$EM(){this._$AL=new Map,this.isUpdatePending=!1}get updateComplete(){return this.getUpdateComplete()}getUpdateComplete(){return this._$ES}shouldUpdate(e){return!0}update(e){this._$Eq&&=this._$Eq.forEach(e=>this._$ET(e,this[e])),this._$EM()}updated(e){}firstUpdated(e){}};$.elementStyles=[],$.shadowRootOptions={mode:"open"},$[m("elementProperties")]=new Map,$[m("finalized")]=new Map,k?.({ReactiveElement:$}),(d.reactiveElementVersions??=[]).push("2.1.1");
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const _=globalThis,y=_.trustedTypes,v=y?y.createPolicy("lit-html",{createHTML:e=>e}):void 0,A="$lit$",S=`lit$${Math.random().toFixed(9).slice(2)}$`,T="?"+S,R=`<${T}>`,E=document,C=()=>E.createComment(""),P=e=>null===e||"object"!=typeof e&&"function"!=typeof e,z=Array.isArray,I="[ \t\n\f\r]",L=/<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g,M=/-->/g,O=/>/g,U=RegExp(`>|${I}(?:([^\\s"'>=/]+)(${I}*=${I}*(?:[^ \t\n\f\r"'\`<>=]|("|')|))|$)`,"g"),B=/'/g,N=/"/g,H=/^(?:script|style|textarea|title)$/i,q=(e=>(t,...s)=>({_$litType$:e,strings:t,values:s}))(1),D=Symbol.for("lit-noChange"),j=Symbol.for("lit-nothing"),Z=new WeakMap,W=E.createTreeWalker(E,129);function Q(e,t){if(!z(e)||!e.hasOwnProperty("raw"))throw Error("invalid template strings array");return void 0!==v?v.createHTML(t):t}const G=(e,t)=>{const s=e.length-1,n=[];let r,i=2===t?"<svg>":3===t?"<math>":"",o=L;for(let t=0;t<s;t++){const s=e[t];let l,a,c=-1,h=0;for(;h<s.length&&(o.lastIndex=h,a=o.exec(s),null!==a);)h=o.lastIndex,o===L?"!--"===a[1]?o=M:void 0!==a[1]?o=O:void 0!==a[2]?(H.test(a[2])&&(r=RegExp("</"+a[2],"g")),o=U):void 0!==a[3]&&(o=U):o===U?">"===a[0]?(o=r??L,c=-1):void 0===a[1]?c=-2:(c=o.lastIndex-a[2].length,l=a[1],o=void 0===a[3]?U:'"'===a[3]?N:B):o===N||o===B?o=U:o===M||o===O?o=L:(o=U,r=void 0);const p=o===U&&e[t+1].startsWith("/>")?" ":"";i+=o===L?s+R:c>=0?(n.push(l),s.slice(0,c)+A+s.slice(c)+S+p):s+S+(-2===c?t:p)}return[Q(e,i+(e[s]||"<?>")+(2===t?"</svg>":3===t?"</math>":"")),n]};let V=class e{constructor({strings:t,_$litType$:s},n){let r;this.parts=[];let i=0,o=0;const l=t.length-1,a=this.parts,[c,h]=G(t,s);if(this.el=e.createElement(c,n),W.currentNode=this.el.content,2===s||3===s){const e=this.el.content.firstChild;e.replaceWith(...e.childNodes)}for(;null!==(r=W.nextNode())&&a.length<l;){if(1===r.nodeType){if(r.hasAttributes())for(const e of r.getAttributeNames())if(e.endsWith(A)){const t=h[o++],s=r.getAttribute(e).split(S),n=/([.?@])?(.*)/.exec(t);a.push({type:1,index:i,name:n[2],strings:s,ctor:"."===n[1]?Y:"?"===n[1]?ee:"@"===n[1]?te:J}),r.removeAttribute(e)}else e.startsWith(S)&&(a.push({type:6,index:i}),r.removeAttribute(e));if(H.test(r.tagName)){const e=r.textContent.split(S),t=e.length-1;if(t>0){r.textContent=y?y.emptyScript:"";for(let s=0;s<t;s++)r.append(e[s],C()),W.nextNode(),a.push({type:2,index:++i});r.append(e[t],C())}}}else if(8===r.nodeType)if(r.data===T)a.push({type:2,index:i});else{let e=-1;for(;-1!==(e=r.data.indexOf(S,e+1));)a.push({type:7,index:i}),e+=S.length-1}i++}}static createElement(e,t){const s=E.createElement("template");return s.innerHTML=e,s}};function F(e,t,s=e,n){if(t===D)return t;let r=void 0!==n?s._$Co?.[n]:s._$Cl;const i=P(t)?void 0:t._$litDirective$;return r?.constructor!==i&&(r?._$AO?.(!1),void 0===i?r=void 0:(r=new i(e),r._$AT(e,s,n)),void 0!==n?(s._$Co??=[])[n]=r:s._$Cl=r),void 0!==r&&(t=F(e,r._$AS(e,t.values),r,n)),t}let K=class{constructor(e,t){this._$AV=[],this._$AN=void 0,this._$AD=e,this._$AM=t}get parentNode(){return this._$AM.parentNode}get _$AU(){return this._$AM._$AU}u(e){const{el:{content:t},parts:s}=this._$AD,n=(e?.creationScope??E).importNode(t,!0);W.currentNode=n;let r=W.nextNode(),i=0,o=0,l=s[0];for(;void 0!==l;){if(i===l.index){let t;2===l.type?t=new X(r,r.nextSibling,this,e):1===l.type?t=new l.ctor(r,l.name,l.strings,this,e):6===l.type&&(t=new se(r,this,e)),this._$AV.push(t),l=s[++o]}i!==l?.index&&(r=W.nextNode(),i++)}return W.currentNode=E,n}p(e){let t=0;for(const s of this._$AV)void 0!==s&&(void 0!==s.strings?(s._$AI(e,s,t),t+=s.strings.length-2):s._$AI(e[t])),t++}};class X{get _$AU(){return this._$AM?._$AU??this._$Cv}constructor(e,t,s,n){this.type=2,this._$AH=j,this._$AN=void 0,this._$AA=e,this._$AB=t,this._$AM=s,this.options=n,this._$Cv=n?.isConnected??!0}get parentNode(){let e=this._$AA.parentNode;const t=this._$AM;return void 0!==t&&11===e?.nodeType&&(e=t.parentNode),e}get startNode(){return this._$AA}get endNode(){return this._$AB}_$AI(e,t=this){e=F(this,e,t),P(e)?e===j||null==e||""===e?(this._$AH!==j&&this._$AR(),this._$AH=j):e!==this._$AH&&e!==D&&this._(e):void 0!==e._$litType$?this.$(e):void 0!==e.nodeType?this.T(e):(e=>z(e)||"function"==typeof e?.[Symbol.iterator])(e)?this.k(e):this._(e)}O(e){return this._$AA.parentNode.insertBefore(e,this._$AB)}T(e){this._$AH!==e&&(this._$AR(),this._$AH=this.O(e))}_(e){this._$AH!==j&&P(this._$AH)?this._$AA.nextSibling.data=e:this.T(E.createTextNode(e)),this._$AH=e}$(e){const{values:t,_$litType$:s}=e,n="number"==typeof s?this._$AC(e):(void 0===s.el&&(s.el=V.createElement(Q(s.h,s.h[0]),this.options)),s);if(this._$AH?._$AD===n)this._$AH.p(t);else{const e=new K(n,this),s=e.u(this.options);e.p(t),this.T(s),this._$AH=e}}_$AC(e){let t=Z.get(e.strings);return void 0===t&&Z.set(e.strings,t=new V(e)),t}k(e){z(this._$AH)||(this._$AH=[],this._$AR());const t=this._$AH;let s,n=0;for(const r of e)n===t.length?t.push(s=new X(this.O(C()),this.O(C()),this,this.options)):s=t[n],s._$AI(r),n++;n<t.length&&(this._$AR(s&&s._$AB.nextSibling,n),t.length=n)}_$AR(e=this._$AA.nextSibling,t){for(this._$AP?.(!1,!0,t);e!==this._$AB;){const t=e.nextSibling;e.remove(),e=t}}setConnected(e){void 0===this._$AM&&(this._$Cv=e,this._$AP?.(e))}}let J=class{get tagName(){return this.element.tagName}get _$AU(){return this._$AM._$AU}constructor(e,t,s,n,r){this.type=1,this._$AH=j,this._$AN=void 0,this.element=e,this.name=t,this._$AM=n,this.options=r,s.length>2||""!==s[0]||""!==s[1]?(this._$AH=Array(s.length-1).fill(new String),this.strings=s):this._$AH=j}_$AI(e,t=this,s,n){const r=this.strings;let i=!1;if(void 0===r)e=F(this,e,t,0),i=!P(e)||e!==this._$AH&&e!==D,i&&(this._$AH=e);else{const n=e;let o,l;for(e=r[0],o=0;o<r.length-1;o++)l=F(this,n[s+o],t,o),l===D&&(l=this._$AH[o]),i||=!P(l)||l!==this._$AH[o],l===j?e=j:e!==j&&(e+=(l??"")+r[o+1]),this._$AH[o]=l}i&&!n&&this.j(e)}j(e){e===j?this.element.removeAttribute(this.name):this.element.setAttribute(this.name,e??"")}};class Y extends J{constructor(){super(...arguments),this.type=3}j(e){this.element[this.name]=e===j?void 0:e}}let ee=class extends J{constructor(){super(...arguments),this.type=4}j(e){this.element.toggleAttribute(this.name,!!e&&e!==j)}},te=class extends J{constructor(e,t,s,n,r){super(e,t,s,n,r),this.type=5}_$AI(e,t=this){if((e=F(this,e,t,0)??j)===D)return;const s=this._$AH,n=e===j&&s!==j||e.capture!==s.capture||e.once!==s.once||e.passive!==s.passive,r=e!==j&&(s===j||n);n&&this.element.removeEventListener(this.name,this,s),r&&this.element.addEventListener(this.name,this,e),this._$AH=e}handleEvent(e){"function"==typeof this._$AH?this._$AH.call(this.options?.host??this.element,e):this._$AH.handleEvent(e)}},se=class{constructor(e,t,s){this.element=e,this.type=6,this._$AN=void 0,this._$AM=t,this.options=s}get _$AU(){return this._$AM._$AU}_$AI(e){F(this,e)}};const ne=_.litHtmlPolyfillSupport;ne?.(V,X),(_.litHtmlVersions??=[]).push("3.3.1");const re=globalThis;
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */let ie=class extends ${constructor(){super(...arguments),this.renderOptions={host:this},this._$Do=void 0}createRenderRoot(){const e=super.createRenderRoot();return this.renderOptions.renderBefore??=e.firstChild,e}update(e){const t=this.render();this.hasUpdated||(this.renderOptions.isConnected=this.isConnected),super.update(e),this._$Do=((e,t,s)=>{const n=s?.renderBefore??t;let r=n._$litPart$;if(void 0===r){const e=s?.renderBefore??null;n._$litPart$=r=new X(t.insertBefore(C(),e),e,void 0,s??{})}return r._$AI(e),r})(t,this.renderRoot,this.renderOptions)}connectedCallback(){super.connectedCallback(),this._$Do?.setConnected(!0)}disconnectedCallback(){super.disconnectedCallback(),this._$Do?.setConnected(!1)}render(){return D}};ie._$litElement$=!0,ie.finalized=!0,re.litElementHydrateSupport?.({LitElement:ie});const oe=re.litElementPolyfillSupport;oe?.({LitElement:ie}),(re.litElementVersions??=[]).push("4.2.1");
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const le={attribute:!0,type:String,converter:x,reflect:!1,hasChanged:b},ae=(e=le,t,s)=>{const{kind:n,metadata:r}=s;let i=globalThis.litPropertyMetadata.get(r);if(void 0===i&&globalThis.litPropertyMetadata.set(r,i=new Map),"setter"===n&&((e=Object.create(e)).wrapped=!0),i.set(s.name,e),"accessor"===n){const{name:n}=s;return{set(s){const r=t.get.call(this);t.set.call(this,s),this.requestUpdate(n,r,e)},init(t){return void 0!==t&&this.C(n,void 0,e,t),t}}}if("setter"===n){const{name:n}=s;return function(s){const r=this[n];t.call(this,s),this.requestUpdate(n,r,e)}}throw Error("Unsupported decorator location: "+n)};
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */function ce(e){return(t,s)=>"object"==typeof s?ae(e,t,s):((e,t,s)=>{const n=t.hasOwnProperty(s);return t.constructor.createProperty(s,e),n?Object.getOwnPropertyDescriptor(t,s):void 0})(e,t,s)}
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */function he(e){return ce({...e,state:!0,attribute:!1})}function pe(){return{async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null}}var ue={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};function de(e){ue=e}var ge={exec:()=>null};function fe(e,t=""){let s="string"==typeof e?e:e.source,n={replace:(e,t)=>{let r="string"==typeof t?t:t.source;return r=r.replace(me.caret,"$1"),s=s.replace(e,r),n},getRegex:()=>new RegExp(s,t)};return n}var ke=(()=>{try{return!!new RegExp("(?<=1)(?<!1)")}catch{return!1}})(),me={codeRemoveIndent:/^(?: {1,4}| {0,3}\t)/gm,outputLinkReplace:/\\([\[\]])/g,indentCodeCompensation:/^(\s+)(?:```)/,beginningSpace:/^\s+/,endingHash:/#$/,startingSpaceChar:/^ /,endingSpaceChar:/ $/,nonSpaceChar:/[^ ]/,newLineCharGlobal:/\n/g,tabCharGlobal:/\t/g,multipleSpaceGlobal:/\s+/g,blankLine:/^[ \t]*$/,doubleBlankLine:/\n[ \t]*\n[ \t]*$/,blockquoteStart:/^ {0,3}>/,blockquoteSetextReplace:/\n {0,3}((?:=+|-+) *)(?=\n|$)/g,blockquoteSetextReplace2:/^ {0,3}>[ \t]?/gm,listReplaceTabs:/^\t+/,listReplaceNesting:/^ {1,4}(?=( {4})*[^ ])/g,listIsTask:/^\[[ xX]\] /,listReplaceTask:/^\[[ xX]\] +/,listTaskCheckbox:/\[[ xX]\]/,anyLine:/\n.*\n/,hrefBrackets:/^<(.*)>$/,tableDelimiter:/[:|]/,tableAlignChars:/^\||\| *$/g,tableRowBlankLine:/\n[ \t]*$/,tableAlignRight:/^ *-+: *$/,tableAlignCenter:/^ *:-+: *$/,tableAlignLeft:/^ *:-+ *$/,startATag:/^<a /i,endATag:/^<\/a>/i,startPreScriptTag:/^<(pre|code|kbd|script)(\s|>)/i,endPreScriptTag:/^<\/(pre|code|kbd|script)(\s|>)/i,startAngleBracket:/^</,endAngleBracket:/>$/,pedanticHrefTitle:/^([^'"]*[^\s])\s+(['"])(.*)\2/,unicodeAlphaNumeric:/[\p{L}\p{N}]/u,escapeTest:/[&<>"']/,escapeReplace:/[&<>"']/g,escapeTestNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/,escapeReplaceNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/g,unescapeTest:/&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi,caret:/(^|[^\[])\^/g,percentDecode:/%25/g,findPipe:/\|/g,splitPipe:/ \|/,slashPipe:/\\\|/g,carriageReturn:/\r\n|\r/g,spaceLine:/^ +$/gm,notSpaceStart:/^\S*/,endingNewline:/\n$/,listItemRegex:e=>new RegExp(`^( {0,3}${e})((?:[\t ][^\\n]*)?(?:\\n|$))`),nextBulletRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ \t][^\\n]*)?(?:\\n|$))`),hrRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`),fencesBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}(?:\`\`\`|~~~)`),headingBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}#`),htmlBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}<(?:[a-z].*>|!--)`,"i")},xe=/^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/,be=/(?:[*+-]|\d{1,9}[.)])/,we=/^(?!bull |blockCode|fences|blockquote|heading|html|table)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html|table))+?)\n {0,3}(=+|-+) *(?:\n+|$)/,$e=fe(we).replace(/bull/g,be).replace(/blockCode/g,/(?: {4}| {0,3}\t)/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).replace(/\|table/g,"").getRegex(),_e=fe(we).replace(/bull/g,be).replace(/blockCode/g,/(?: {4}| {0,3}\t)/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).replace(/table/g,/ {0,3}\|?(?:[:\- ]*\|)+[\:\- ]*\n/).getRegex(),ye=/^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,ve=/(?!\s*\])(?:\\[\s\S]|[^\[\]\\])+/,Ae=fe(/^ {0,3}\[(label)\]: *(?:\n[ \t]*)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n[ \t]*)?| *\n[ \t]*)(title))? *(?:\n+|$)/).replace("label",ve).replace("title",/(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(),Se=fe(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g,be).getRegex(),Te="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",Re=/<!--(?:-?>|[\s\S]*?(?:-->|$))/,Ee=fe("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$))","i").replace("comment",Re).replace("tag",Te).replace("attribute",/ +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(),Ce=fe(ye).replace("hr",xe).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",Te).getRegex(),Pe={blockquote:fe(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph",Ce).getRegex(),code:/^((?: {4}| {0,3}\t)[^\n]+(?:\n(?:[ \t]*(?:\n|$))*)?)+/,def:Ae,fences:/^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/,heading:/^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,hr:xe,html:Ee,lheading:$e,list:Se,newline:/^(?:[ \t]*(?:\n|$))+/,paragraph:Ce,table:ge,text:/^[^\n]+/},ze=fe("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr",xe).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("blockquote"," {0,3}>").replace("code","(?: {4}| {0,3}\t)[^\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",Te).getRegex(),Ie={...Pe,lheading:_e,table:ze,paragraph:fe(ye).replace("hr",xe).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("table",ze).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",Te).getRegex()},Le={...Pe,html:fe("^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))").replace("comment",Re).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),def:/^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,heading:/^(#{1,6})(.*)(?:\n+|$)/,fences:ge,lheading:/^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,paragraph:fe(ye).replace("hr",xe).replace("heading"," *#{1,6} *[^\n]").replace("lheading",$e).replace("|table","").replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").replace("|tag","").getRegex()},Me=/^( {2,}|\\)\n(?!\s*$)/,Oe=/[\p{P}\p{S}]/u,Ue=/[\s\p{P}\p{S}]/u,Be=/[^\s\p{P}\p{S}]/u,Ne=fe(/^((?![*_])punctSpace)/,"u").replace(/punctSpace/g,Ue).getRegex(),He=/(?!~)[\p{P}\p{S}]/u,qe=fe(/link|precode-code|html/,"g").replace("link",/\[(?:[^\[\]`]|(?<a>`+)[^`]+\k<a>(?!`))*?\]\((?:\\[\s\S]|[^\\\(\)]|\((?:\\[\s\S]|[^\\\(\)])*\))*\)/).replace("precode-",ke?"(?<!`)()":"(^^|[^`])").replace("code",/(?<b>`+)[^`]+\k<b>(?!`)/).replace("html",/<(?! )[^<>]*?>/).getRegex(),De=/^(?:\*+(?:((?!\*)punct)|[^\s*]))|^_+(?:((?!_)punct)|([^\s_]))/,je=fe(De,"u").replace(/punct/g,Oe).getRegex(),Ze=fe(De,"u").replace(/punct/g,He).getRegex(),We="^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)punct(\\*+)(?=[\\s]|$)|notPunctSpace(\\*+)(?!\\*)(?=punctSpace|$)|(?!\\*)punctSpace(\\*+)(?=notPunctSpace)|[\\s](\\*+)(?!\\*)(?=punct)|(?!\\*)punct(\\*+)(?!\\*)(?=punct)|notPunctSpace(\\*+)(?=notPunctSpace)",Qe=fe(We,"gu").replace(/notPunctSpace/g,Be).replace(/punctSpace/g,Ue).replace(/punct/g,Oe).getRegex(),Ge=fe(We,"gu").replace(/notPunctSpace/g,/(?:[^\s\p{P}\p{S}]|~)/u).replace(/punctSpace/g,/(?!~)[\s\p{P}\p{S}]/u).replace(/punct/g,He).getRegex(),Ve=fe("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)punct(_+)(?=[\\s]|$)|notPunctSpace(_+)(?!_)(?=punctSpace|$)|(?!_)punctSpace(_+)(?=notPunctSpace)|[\\s](_+)(?!_)(?=punct)|(?!_)punct(_+)(?!_)(?=punct)","gu").replace(/notPunctSpace/g,Be).replace(/punctSpace/g,Ue).replace(/punct/g,Oe).getRegex(),Fe=fe(/\\(punct)/,"gu").replace(/punct/g,Oe).getRegex(),Ke=fe(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme",/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email",/[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(),Xe=fe(Re).replace("(?:--\x3e|$)","--\x3e").getRegex(),Je=fe("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment",Xe).replace("attribute",/\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(),Ye=/(?:\[(?:\\[\s\S]|[^\[\]\\])*\]|\\[\s\S]|`+[^`]*?`+(?!`)|[^\[\]\\`])*?/,et=fe(/^!?\[(label)\]\(\s*(href)(?:(?:[ \t]*(?:\n[ \t]*)?)(title))?\s*\)/).replace("label",Ye).replace("href",/<(?:\\.|[^\n<>\\])+>|[^ \t\n\x00-\x1f]*/).replace("title",/"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(),tt=fe(/^!?\[(label)\]\[(ref)\]/).replace("label",Ye).replace("ref",ve).getRegex(),st=fe(/^!?\[(ref)\](?:\[\])?/).replace("ref",ve).getRegex(),nt=/[hH][tT][tT][pP][sS]?|[fF][tT][pP]/,rt={_backpedal:ge,anyPunctuation:Fe,autolink:Ke,blockSkip:qe,br:Me,code:/^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,del:ge,emStrongLDelim:je,emStrongRDelimAst:Qe,emStrongRDelimUnd:Ve,escape:/^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,link:et,nolink:st,punctuation:Ne,reflink:tt,reflinkSearch:fe("reflink|nolink(?!\\()","g").replace("reflink",tt).replace("nolink",st).getRegex(),tag:Je,text:/^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,url:ge},it={...rt,link:fe(/^!?\[(label)\]\((.*?)\)/).replace("label",Ye).getRegex(),reflink:fe(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label",Ye).getRegex()},ot={...rt,emStrongRDelimAst:Ge,emStrongLDelim:Ze,url:fe(/^((?:protocol):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/).replace("protocol",nt).replace("email",/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),_backpedal:/(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,del:/^(~~?)(?=[^\s~])((?:\\[\s\S]|[^\\])*?(?:\\[\s\S]|[^\s~\\]))\1(?=[^~]|$)/,text:fe(/^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|protocol:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/).replace("protocol",nt).getRegex()},lt={...ot,br:fe(Me).replace("{2,}","*").getRegex(),text:fe(ot.text).replace("\\b_","\\b_| {2,}\\n").replace(/\{2,\}/g,"*").getRegex()},at={normal:Pe,gfm:Ie,pedantic:Le},ct={normal:rt,gfm:ot,breaks:lt,pedantic:it},ht={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},pt=e=>ht[e];function ut(e,t){if(t){if(me.escapeTest.test(e))return e.replace(me.escapeReplace,pt)}else if(me.escapeTestNoEncode.test(e))return e.replace(me.escapeReplaceNoEncode,pt);return e}function dt(e){try{e=encodeURI(e).replace(me.percentDecode,"%")}catch{return null}return e}function gt(e,t){let s=e.replace(me.findPipe,(e,t,s)=>{let n=!1,r=t;for(;--r>=0&&"\\"===s[r];)n=!n;return n?"|":" |"}),n=s.split(me.splitPipe),r=0;if(n[0].trim()||n.shift(),n.length>0&&!n.at(-1)?.trim()&&n.pop(),t)if(n.length>t)n.splice(t);else for(;n.length<t;)n.push("");for(;r<n.length;r++)n[r]=n[r].trim().replace(me.slashPipe,"|");return n}function ft(e,t,s){let n=e.length;if(0===n)return"";let r=0;for(;r<n;){if(e.charAt(n-r-1)!==t)break;r++}return e.slice(0,n-r)}function kt(e,t,s,n,r){let i=t.href,o=t.title||null,l=e[1].replace(r.other.outputLinkReplace,"$1");n.state.inLink=!0;let a={type:"!"===e[0].charAt(0)?"image":"link",raw:s,href:i,title:o,text:l,tokens:n.inlineTokens(l)};return n.state.inLink=!1,a}var mt=class{options;rules;lexer;constructor(e){this.options=e||ue}space(e){let t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return{type:"space",raw:t[0]}}code(e){let t=this.rules.block.code.exec(e);if(t){let e=t[0].replace(this.rules.other.codeRemoveIndent,"");return{type:"code",raw:t[0],codeBlockStyle:"indented",text:this.options.pedantic?e:ft(e,"\n")}}}fences(e){let t=this.rules.block.fences.exec(e);if(t){let e=t[0],s=function(e,t,s){let n=e.match(s.other.indentCodeCompensation);if(null===n)return t;let r=n[1];return t.split("\n").map(e=>{let t=e.match(s.other.beginningSpace);if(null===t)return e;let[n]=t;return n.length>=r.length?e.slice(r.length):e}).join("\n")}(e,t[3]||"",this.rules);return{type:"code",raw:e,lang:t[2]?t[2].trim().replace(this.rules.inline.anyPunctuation,"$1"):t[2],text:s}}}heading(e){let t=this.rules.block.heading.exec(e);if(t){let e=t[2].trim();if(this.rules.other.endingHash.test(e)){let t=ft(e,"#");(this.options.pedantic||!t||this.rules.other.endingSpaceChar.test(t))&&(e=t.trim())}return{type:"heading",raw:t[0],depth:t[1].length,text:e,tokens:this.lexer.inline(e)}}}hr(e){let t=this.rules.block.hr.exec(e);if(t)return{type:"hr",raw:ft(t[0],"\n")}}blockquote(e){let t=this.rules.block.blockquote.exec(e);if(t){let e=ft(t[0],"\n").split("\n"),s="",n="",r=[];for(;e.length>0;){let t,i=!1,o=[];for(t=0;t<e.length;t++)if(this.rules.other.blockquoteStart.test(e[t]))o.push(e[t]),i=!0;else{if(i)break;o.push(e[t])}e=e.slice(t);let l=o.join("\n"),a=l.replace(this.rules.other.blockquoteSetextReplace,"\n    $1").replace(this.rules.other.blockquoteSetextReplace2,"");s=s?`${s}\n${l}`:l,n=n?`${n}\n${a}`:a;let c=this.lexer.state.top;if(this.lexer.state.top=!0,this.lexer.blockTokens(a,r,!0),this.lexer.state.top=c,0===e.length)break;let h=r.at(-1);if("code"===h?.type)break;if("blockquote"===h?.type){let t=h,i=t.raw+"\n"+e.join("\n"),o=this.blockquote(i);r[r.length-1]=o,s=s.substring(0,s.length-t.raw.length)+o.raw,n=n.substring(0,n.length-t.text.length)+o.text;break}if("list"===h?.type){let t=h,i=t.raw+"\n"+e.join("\n"),o=this.list(i);r[r.length-1]=o,s=s.substring(0,s.length-h.raw.length)+o.raw,n=n.substring(0,n.length-t.raw.length)+o.raw,e=i.substring(r.at(-1).raw.length).split("\n");continue}}return{type:"blockquote",raw:s,tokens:r,text:n}}}list(e){let t=this.rules.block.list.exec(e);if(t){let s=t[1].trim(),n=s.length>1,r={type:"list",raw:"",ordered:n,start:n?+s.slice(0,-1):"",loose:!1,items:[]};s=n?`\\d{1,9}\\${s.slice(-1)}`:`\\${s}`,this.options.pedantic&&(s=n?s:"[*+-]");let i=this.rules.other.listItemRegex(s),o=!1;for(;e;){let s=!1,n="",l="";if(!(t=i.exec(e))||this.rules.block.hr.test(e))break;n=t[0],e=e.substring(n.length);let a=t[2].split("\n",1)[0].replace(this.rules.other.listReplaceTabs,e=>" ".repeat(3*e.length)),c=e.split("\n",1)[0],h=!a.trim(),p=0;if(this.options.pedantic?(p=2,l=a.trimStart()):h?p=t[1].length+1:(p=t[2].search(this.rules.other.nonSpaceChar),p=p>4?1:p,l=a.slice(p),p+=t[1].length),h&&this.rules.other.blankLine.test(c)&&(n+=c+"\n",e=e.substring(c.length+1),s=!0),!s){let t=this.rules.other.nextBulletRegex(p),s=this.rules.other.hrRegex(p),r=this.rules.other.fencesBeginRegex(p),i=this.rules.other.headingBeginRegex(p),o=this.rules.other.htmlBeginRegex(p);for(;e;){let u,d=e.split("\n",1)[0];if(c=d,this.options.pedantic?(c=c.replace(this.rules.other.listReplaceNesting,"  "),u=c):u=c.replace(this.rules.other.tabCharGlobal,"    "),r.test(c)||i.test(c)||o.test(c)||t.test(c)||s.test(c))break;if(u.search(this.rules.other.nonSpaceChar)>=p||!c.trim())l+="\n"+u.slice(p);else{if(h||a.replace(this.rules.other.tabCharGlobal,"    ").search(this.rules.other.nonSpaceChar)>=4||r.test(a)||i.test(a)||s.test(a))break;l+="\n"+c}!h&&!c.trim()&&(h=!0),n+=d+"\n",e=e.substring(d.length+1),a=u.slice(p)}}r.loose||(o?r.loose=!0:this.rules.other.doubleBlankLine.test(n)&&(o=!0));let u=null;this.options.gfm&&(u=this.rules.other.listIsTask.exec(l),u&&(l=l.replace(this.rules.other.listReplaceTask,""))),r.items.push({type:"list_item",raw:n,task:!!u,loose:!1,text:l,tokens:[]}),r.raw+=n}let l=r.items.at(-1);if(!l)return;l.raw=l.raw.trimEnd(),l.text=l.text.trimEnd(),r.raw=r.raw.trimEnd();for(let e of r.items){if(this.lexer.state.top=!1,e.tokens=this.lexer.blockTokens(e.text,[]),e.task){let t=this.rules.other.listTaskCheckbox.exec(e.raw);if(t){let s={type:"checkbox",raw:t[0]+" ",checked:"[ ]"!==t[0]};e.checked=s.checked,r.loose?e.tokens[0]&&["paragraph","text"].includes(e.tokens[0].type)&&"tokens"in e.tokens[0]&&e.tokens[0].tokens?(e.tokens[0].raw=s.raw+e.tokens[0].raw,e.tokens[0].text=s.raw+e.tokens[0].text,e.tokens[0].tokens.unshift(s)):e.tokens.unshift({type:"paragraph",raw:s.raw,text:s.raw,tokens:[s]}):e.tokens.unshift(s)}}if(!r.loose){let t=e.tokens.filter(e=>"space"===e.type),s=t.length>0&&t.some(e=>this.rules.other.anyLine.test(e.raw));r.loose=s}}if(r.loose)for(let e of r.items){e.loose=!0;for(let t of e.tokens)"text"===t.type&&(t.type="paragraph")}return r}}html(e){let t=this.rules.block.html.exec(e);if(t)return{type:"html",block:!0,raw:t[0],pre:"pre"===t[1]||"script"===t[1]||"style"===t[1],text:t[0]}}def(e){let t=this.rules.block.def.exec(e);if(t){let e=t[1].toLowerCase().replace(this.rules.other.multipleSpaceGlobal," "),s=t[2]?t[2].replace(this.rules.other.hrefBrackets,"$1").replace(this.rules.inline.anyPunctuation,"$1"):"",n=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline.anyPunctuation,"$1"):t[3];return{type:"def",tag:e,raw:t[0],href:s,title:n}}}table(e){let t=this.rules.block.table.exec(e);if(!t||!this.rules.other.tableDelimiter.test(t[2]))return;let s=gt(t[1]),n=t[2].replace(this.rules.other.tableAlignChars,"").split("|"),r=t[3]?.trim()?t[3].replace(this.rules.other.tableRowBlankLine,"").split("\n"):[],i={type:"table",raw:t[0],header:[],align:[],rows:[]};if(s.length===n.length){for(let e of n)this.rules.other.tableAlignRight.test(e)?i.align.push("right"):this.rules.other.tableAlignCenter.test(e)?i.align.push("center"):this.rules.other.tableAlignLeft.test(e)?i.align.push("left"):i.align.push(null);for(let e=0;e<s.length;e++)i.header.push({text:s[e],tokens:this.lexer.inline(s[e]),header:!0,align:i.align[e]});for(let e of r)i.rows.push(gt(e,i.header.length).map((e,t)=>({text:e,tokens:this.lexer.inline(e),header:!1,align:i.align[t]})));return i}}lheading(e){let t=this.rules.block.lheading.exec(e);if(t)return{type:"heading",raw:t[0],depth:"="===t[2].charAt(0)?1:2,text:t[1],tokens:this.lexer.inline(t[1])}}paragraph(e){let t=this.rules.block.paragraph.exec(e);if(t){let e="\n"===t[1].charAt(t[1].length-1)?t[1].slice(0,-1):t[1];return{type:"paragraph",raw:t[0],text:e,tokens:this.lexer.inline(e)}}}text(e){let t=this.rules.block.text.exec(e);if(t)return{type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){let t=this.rules.inline.escape.exec(e);if(t)return{type:"escape",raw:t[0],text:t[1]}}tag(e){let t=this.rules.inline.tag.exec(e);if(t)return!this.lexer.state.inLink&&this.rules.other.startATag.test(t[0])?this.lexer.state.inLink=!0:this.lexer.state.inLink&&this.rules.other.endATag.test(t[0])&&(this.lexer.state.inLink=!1),!this.lexer.state.inRawBlock&&this.rules.other.startPreScriptTag.test(t[0])?this.lexer.state.inRawBlock=!0:this.lexer.state.inRawBlock&&this.rules.other.endPreScriptTag.test(t[0])&&(this.lexer.state.inRawBlock=!1),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:!1,text:t[0]}}link(e){let t=this.rules.inline.link.exec(e);if(t){let e=t[2].trim();if(!this.options.pedantic&&this.rules.other.startAngleBracket.test(e)){if(!this.rules.other.endAngleBracket.test(e))return;let t=ft(e.slice(0,-1),"\\");if((e.length-t.length)%2==0)return}else{let e=function(e,t){if(-1===e.indexOf(t[1]))return-1;let s=0;for(let n=0;n<e.length;n++)if("\\"===e[n])n++;else if(e[n]===t[0])s++;else if(e[n]===t[1]&&(s--,s<0))return n;return s>0?-2:-1}(t[2],"()");if(-2===e)return;if(e>-1){let s=(0===t[0].indexOf("!")?5:4)+t[1].length+e;t[2]=t[2].substring(0,e),t[0]=t[0].substring(0,s).trim(),t[3]=""}}let s=t[2],n="";if(this.options.pedantic){let e=this.rules.other.pedanticHrefTitle.exec(s);e&&(s=e[1],n=e[3])}else n=t[3]?t[3].slice(1,-1):"";return s=s.trim(),this.rules.other.startAngleBracket.test(s)&&(s=this.options.pedantic&&!this.rules.other.endAngleBracket.test(e)?s.slice(1):s.slice(1,-1)),kt(t,{href:s&&s.replace(this.rules.inline.anyPunctuation,"$1"),title:n&&n.replace(this.rules.inline.anyPunctuation,"$1")},t[0],this.lexer,this.rules)}}reflink(e,t){let s;if((s=this.rules.inline.reflink.exec(e))||(s=this.rules.inline.nolink.exec(e))){let e=t[(s[2]||s[1]).replace(this.rules.other.multipleSpaceGlobal," ").toLowerCase()];if(!e){let e=s[0].charAt(0);return{type:"text",raw:e,text:e}}return kt(s,e,s[0],this.lexer,this.rules)}}emStrong(e,t,s=""){let n=this.rules.inline.emStrongLDelim.exec(e);if(!(!n||n[3]&&s.match(this.rules.other.unicodeAlphaNumeric))&&(!n[1]&&!n[2]||!s||this.rules.inline.punctuation.exec(s))){let s,r,i=[...n[0]].length-1,o=i,l=0,a="*"===n[0][0]?this.rules.inline.emStrongRDelimAst:this.rules.inline.emStrongRDelimUnd;for(a.lastIndex=0,t=t.slice(-1*e.length+i);null!=(n=a.exec(t));){if(s=n[1]||n[2]||n[3]||n[4]||n[5]||n[6],!s)continue;if(r=[...s].length,n[3]||n[4]){o+=r;continue}if((n[5]||n[6])&&i%3&&!((i+r)%3)){l+=r;continue}if(o-=r,o>0)continue;r=Math.min(r,r+o+l);let t=[...n[0]][0].length,a=e.slice(0,i+n.index+t+r);if(Math.min(i,r)%2){let e=a.slice(1,-1);return{type:"em",raw:a,text:e,tokens:this.lexer.inlineTokens(e)}}let c=a.slice(2,-2);return{type:"strong",raw:a,text:c,tokens:this.lexer.inlineTokens(c)}}}}codespan(e){let t=this.rules.inline.code.exec(e);if(t){let e=t[2].replace(this.rules.other.newLineCharGlobal," "),s=this.rules.other.nonSpaceChar.test(e),n=this.rules.other.startingSpaceChar.test(e)&&this.rules.other.endingSpaceChar.test(e);return s&&n&&(e=e.substring(1,e.length-1)),{type:"codespan",raw:t[0],text:e}}}br(e){let t=this.rules.inline.br.exec(e);if(t)return{type:"br",raw:t[0]}}del(e){let t=this.rules.inline.del.exec(e);if(t)return{type:"del",raw:t[0],text:t[2],tokens:this.lexer.inlineTokens(t[2])}}autolink(e){let t=this.rules.inline.autolink.exec(e);if(t){let e,s;return"@"===t[2]?(e=t[1],s="mailto:"+e):(e=t[1],s=e),{type:"link",raw:t[0],text:e,href:s,tokens:[{type:"text",raw:e,text:e}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let e,s;if("@"===t[2])e=t[0],s="mailto:"+e;else{let n;do{n=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])?.[0]??""}while(n!==t[0]);e=t[0],s="www."===t[1]?"http://"+t[0]:t[0]}return{type:"link",raw:t[0],text:e,href:s,tokens:[{type:"text",raw:e,text:e}]}}}inlineText(e){let t=this.rules.inline.text.exec(e);if(t){let e=this.lexer.state.inRawBlock;return{type:"text",raw:t[0],text:t[0],escaped:e}}}},xt=class e{tokens;options;state;tokenizer;inlineQueue;constructor(e){this.tokens=[],this.tokens.links=Object.create(null),this.options=e||ue,this.options.tokenizer=this.options.tokenizer||new mt,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:!1,inRawBlock:!1,top:!0};let t={other:me,block:at.normal,inline:ct.normal};this.options.pedantic?(t.block=at.pedantic,t.inline=ct.pedantic):this.options.gfm&&(t.block=at.gfm,this.options.breaks?t.inline=ct.breaks:t.inline=ct.gfm),this.tokenizer.rules=t}static get rules(){return{block:at,inline:ct}}static lex(t,s){return new e(s).lex(t)}static lexInline(t,s){return new e(s).inlineTokens(t)}lex(e){e=e.replace(me.carriageReturn,"\n"),this.blockTokens(e,this.tokens);for(let e=0;e<this.inlineQueue.length;e++){let t=this.inlineQueue[e];this.inlineTokens(t.src,t.tokens)}return this.inlineQueue=[],this.tokens}blockTokens(e,t=[],s=!1){for(this.options.pedantic&&(e=e.replace(me.tabCharGlobal,"    ").replace(me.spaceLine,""));e;){let n;if(this.options.extensions?.block?.some(s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))continue;if(n=this.tokenizer.space(e)){e=e.substring(n.raw.length);let s=t.at(-1);1===n.raw.length&&void 0!==s?s.raw+="\n":t.push(n);continue}if(n=this.tokenizer.code(e)){e=e.substring(n.raw.length);let s=t.at(-1);"paragraph"===s?.type||"text"===s?.type?(s.raw+=(s.raw.endsWith("\n")?"":"\n")+n.raw,s.text+="\n"+n.text,this.inlineQueue.at(-1).src=s.text):t.push(n);continue}if(n=this.tokenizer.fences(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.heading(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.hr(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.blockquote(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.list(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.html(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.def(e)){e=e.substring(n.raw.length);let s=t.at(-1);"paragraph"===s?.type||"text"===s?.type?(s.raw+=(s.raw.endsWith("\n")?"":"\n")+n.raw,s.text+="\n"+n.raw,this.inlineQueue.at(-1).src=s.text):this.tokens.links[n.tag]||(this.tokens.links[n.tag]={href:n.href,title:n.title},t.push(n));continue}if(n=this.tokenizer.table(e)){e=e.substring(n.raw.length),t.push(n);continue}if(n=this.tokenizer.lheading(e)){e=e.substring(n.raw.length),t.push(n);continue}let r=e;if(this.options.extensions?.startBlock){let t,s=1/0,n=e.slice(1);this.options.extensions.startBlock.forEach(e=>{t=e.call({lexer:this},n),"number"==typeof t&&t>=0&&(s=Math.min(s,t))}),s<1/0&&s>=0&&(r=e.substring(0,s+1))}if(this.state.top&&(n=this.tokenizer.paragraph(r))){let i=t.at(-1);s&&"paragraph"===i?.type?(i.raw+=(i.raw.endsWith("\n")?"":"\n")+n.raw,i.text+="\n"+n.text,this.inlineQueue.pop(),this.inlineQueue.at(-1).src=i.text):t.push(n),s=r.length!==e.length,e=e.substring(n.raw.length);continue}if(n=this.tokenizer.text(e)){e=e.substring(n.raw.length);let s=t.at(-1);"text"===s?.type?(s.raw+=(s.raw.endsWith("\n")?"":"\n")+n.raw,s.text+="\n"+n.text,this.inlineQueue.pop(),this.inlineQueue.at(-1).src=s.text):t.push(n);continue}if(e){let t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return this.state.top=!0,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){let s,n=e,r=null;if(this.tokens.links){let e=Object.keys(this.tokens.links);if(e.length>0)for(;null!=(r=this.tokenizer.rules.inline.reflinkSearch.exec(n));)e.includes(r[0].slice(r[0].lastIndexOf("[")+1,-1))&&(n=n.slice(0,r.index)+"["+"a".repeat(r[0].length-2)+"]"+n.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))}for(;null!=(r=this.tokenizer.rules.inline.anyPunctuation.exec(n));)n=n.slice(0,r.index)+"++"+n.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);for(;null!=(r=this.tokenizer.rules.inline.blockSkip.exec(n));)s=r[2]?r[2].length:0,n=n.slice(0,r.index+s)+"["+"a".repeat(r[0].length-s-2)+"]"+n.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);n=this.options.hooks?.emStrongMask?.call({lexer:this},n)??n;let i=!1,o="";for(;e;){let s;if(i||(o=""),i=!1,this.options.extensions?.inline?.some(n=>!!(s=n.call({lexer:this},e,t))&&(e=e.substring(s.raw.length),t.push(s),!0)))continue;if(s=this.tokenizer.escape(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.tag(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.link(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.reflink(e,this.tokens.links)){e=e.substring(s.raw.length);let n=t.at(-1);"text"===s.type&&"text"===n?.type?(n.raw+=s.raw,n.text+=s.text):t.push(s);continue}if(s=this.tokenizer.emStrong(e,n,o)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.codespan(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.br(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.del(e)){e=e.substring(s.raw.length),t.push(s);continue}if(s=this.tokenizer.autolink(e)){e=e.substring(s.raw.length),t.push(s);continue}if(!this.state.inLink&&(s=this.tokenizer.url(e))){e=e.substring(s.raw.length),t.push(s);continue}let r=e;if(this.options.extensions?.startInline){let t,s=1/0,n=e.slice(1);this.options.extensions.startInline.forEach(e=>{t=e.call({lexer:this},n),"number"==typeof t&&t>=0&&(s=Math.min(s,t))}),s<1/0&&s>=0&&(r=e.substring(0,s+1))}if(s=this.tokenizer.inlineText(r)){e=e.substring(s.raw.length),"_"!==s.raw.slice(-1)&&(o=s.raw.slice(-1)),i=!0;let n=t.at(-1);"text"===n?.type?(n.raw+=s.raw,n.text+=s.text):t.push(s);continue}if(e){let t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return t}},bt=class{options;parser;constructor(e){this.options=e||ue}space(e){return""}code({text:e,lang:t,escaped:s}){let n=(t||"").match(me.notSpaceStart)?.[0],r=e.replace(me.endingNewline,"")+"\n";return n?'<pre><code class="language-'+ut(n)+'">'+(s?r:ut(r,!0))+"</code></pre>\n":"<pre><code>"+(s?r:ut(r,!0))+"</code></pre>\n"}blockquote({tokens:e}){return`<blockquote>\n${this.parser.parse(e)}</blockquote>\n`}html({text:e}){return e}def(e){return""}heading({tokens:e,depth:t}){return`<h${t}>${this.parser.parseInline(e)}</h${t}>\n`}hr(e){return"<hr>\n"}list(e){let t=e.ordered,s=e.start,n="";for(let t=0;t<e.items.length;t++){let s=e.items[t];n+=this.listitem(s)}let r=t?"ol":"ul";return"<"+r+(t&&1!==s?' start="'+s+'"':"")+">\n"+n+"</"+r+">\n"}listitem(e){return`<li>${this.parser.parse(e.tokens)}</li>\n`}checkbox({checked:e}){return"<input "+(e?'checked="" ':"")+'disabled="" type="checkbox"> '}paragraph({tokens:e}){return`<p>${this.parser.parseInline(e)}</p>\n`}table(e){let t="",s="";for(let t=0;t<e.header.length;t++)s+=this.tablecell(e.header[t]);t+=this.tablerow({text:s});let n="";for(let t=0;t<e.rows.length;t++){let r=e.rows[t];s="";for(let e=0;e<r.length;e++)s+=this.tablecell(r[e]);n+=this.tablerow({text:s})}return n&&(n=`<tbody>${n}</tbody>`),"<table>\n<thead>\n"+t+"</thead>\n"+n+"</table>\n"}tablerow({text:e}){return`<tr>\n${e}</tr>\n`}tablecell(e){let t=this.parser.parseInline(e.tokens),s=e.header?"th":"td";return(e.align?`<${s} align="${e.align}">`:`<${s}>`)+t+`</${s}>\n`}strong({tokens:e}){return`<strong>${this.parser.parseInline(e)}</strong>`}em({tokens:e}){return`<em>${this.parser.parseInline(e)}</em>`}codespan({text:e}){return`<code>${ut(e,!0)}</code>`}br(e){return"<br>"}del({tokens:e}){return`<del>${this.parser.parseInline(e)}</del>`}link({href:e,title:t,tokens:s}){let n=this.parser.parseInline(s),r=dt(e);if(null===r)return n;let i='<a href="'+(e=r)+'"';return t&&(i+=' title="'+ut(t)+'"'),i+=">"+n+"</a>",i}image({href:e,title:t,text:s,tokens:n}){n&&(s=this.parser.parseInline(n,this.parser.textRenderer));let r=dt(e);if(null===r)return ut(s);let i=`<img src="${e=r}" alt="${s}"`;return t&&(i+=` title="${ut(t)}"`),i+=">",i}text(e){return"tokens"in e&&e.tokens?this.parser.parseInline(e.tokens):"escaped"in e&&e.escaped?e.text:ut(e.text)}},wt=class{strong({text:e}){return e}em({text:e}){return e}codespan({text:e}){return e}del({text:e}){return e}html({text:e}){return e}text({text:e}){return e}link({text:e}){return""+e}image({text:e}){return""+e}br(){return""}checkbox({raw:e}){return e}},$t=class e{options;renderer;textRenderer;constructor(e){this.options=e||ue,this.options.renderer=this.options.renderer||new bt,this.renderer=this.options.renderer,this.renderer.options=this.options,this.renderer.parser=this,this.textRenderer=new wt}static parse(t,s){return new e(s).parse(t)}static parseInline(t,s){return new e(s).parseInline(t)}parse(e){let t="";for(let s=0;s<e.length;s++){let n=e[s];if(this.options.extensions?.renderers?.[n.type]){let e=n,s=this.options.extensions.renderers[e.type].call({parser:this},e);if(!1!==s||!["space","hr","heading","code","table","blockquote","list","html","def","paragraph","text"].includes(e.type)){t+=s||"";continue}}let r=n;switch(r.type){case"space":t+=this.renderer.space(r);break;case"hr":t+=this.renderer.hr(r);break;case"heading":t+=this.renderer.heading(r);break;case"code":t+=this.renderer.code(r);break;case"table":t+=this.renderer.table(r);break;case"blockquote":t+=this.renderer.blockquote(r);break;case"list":t+=this.renderer.list(r);break;case"checkbox":t+=this.renderer.checkbox(r);break;case"html":t+=this.renderer.html(r);break;case"def":t+=this.renderer.def(r);break;case"paragraph":t+=this.renderer.paragraph(r);break;case"text":t+=this.renderer.text(r);break;default:{let e='Token with "'+r.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return t}parseInline(e,t=this.renderer){let s="";for(let n=0;n<e.length;n++){let r=e[n];if(this.options.extensions?.renderers?.[r.type]){let e=this.options.extensions.renderers[r.type].call({parser:this},r);if(!1!==e||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){s+=e||"";continue}}let i=r;switch(i.type){case"escape":case"text":s+=t.text(i);break;case"html":s+=t.html(i);break;case"link":s+=t.link(i);break;case"image":s+=t.image(i);break;case"checkbox":s+=t.checkbox(i);break;case"strong":s+=t.strong(i);break;case"em":s+=t.em(i);break;case"codespan":s+=t.codespan(i);break;case"br":s+=t.br(i);break;case"del":s+=t.del(i);break;default:{let e='Token with "'+i.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return s}},_t=class{options;block;constructor(e){this.options=e||ue}static passThroughHooks=new Set(["preprocess","postprocess","processAllTokens","emStrongMask"]);static passThroughHooksRespectAsync=new Set(["preprocess","postprocess","processAllTokens"]);preprocess(e){return e}postprocess(e){return e}processAllTokens(e){return e}emStrongMask(e){return e}provideLexer(){return this.block?xt.lex:xt.lexInline}provideParser(){return this.block?$t.parse:$t.parseInline}},yt=new class{defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};options=this.setOptions;parse=this.parseMarkdown(!0);parseInline=this.parseMarkdown(!1);Parser=$t;Renderer=bt;TextRenderer=wt;Lexer=xt;Tokenizer=mt;Hooks=_t;constructor(...e){this.use(...e)}walkTokens(e,t){let s=[];for(let n of e)switch(s=s.concat(t.call(this,n)),n.type){case"table":{let e=n;for(let n of e.header)s=s.concat(this.walkTokens(n.tokens,t));for(let n of e.rows)for(let e of n)s=s.concat(this.walkTokens(e.tokens,t));break}case"list":{let e=n;s=s.concat(this.walkTokens(e.items,t));break}default:{let e=n;this.defaults.extensions?.childTokens?.[e.type]?this.defaults.extensions.childTokens[e.type].forEach(n=>{let r=e[n].flat(1/0);s=s.concat(this.walkTokens(r,t))}):e.tokens&&(s=s.concat(this.walkTokens(e.tokens,t)))}}return s}use(...e){let t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach(e=>{let s={...e};if(s.async=this.defaults.async||s.async||!1,e.extensions&&(e.extensions.forEach(e=>{if(!e.name)throw new Error("extension name required");if("renderer"in e){let s=t.renderers[e.name];t.renderers[e.name]=s?function(...t){let n=e.renderer.apply(this,t);return!1===n&&(n=s.apply(this,t)),n}:e.renderer}if("tokenizer"in e){if(!e.level||"block"!==e.level&&"inline"!==e.level)throw new Error("extension level must be 'block' or 'inline'");let s=t[e.level];s?s.unshift(e.tokenizer):t[e.level]=[e.tokenizer],e.start&&("block"===e.level?t.startBlock?t.startBlock.push(e.start):t.startBlock=[e.start]:"inline"===e.level&&(t.startInline?t.startInline.push(e.start):t.startInline=[e.start]))}"childTokens"in e&&e.childTokens&&(t.childTokens[e.name]=e.childTokens)}),s.extensions=t),e.renderer){let t=this.defaults.renderer||new bt(this.defaults);for(let s in e.renderer){if(!(s in t))throw new Error(`renderer '${s}' does not exist`);if(["options","parser"].includes(s))continue;let n=s,r=e.renderer[n],i=t[n];t[n]=(...e)=>{let s=r.apply(t,e);return!1===s&&(s=i.apply(t,e)),s||""}}s.renderer=t}if(e.tokenizer){let t=this.defaults.tokenizer||new mt(this.defaults);for(let s in e.tokenizer){if(!(s in t))throw new Error(`tokenizer '${s}' does not exist`);if(["options","rules","lexer"].includes(s))continue;let n=s,r=e.tokenizer[n],i=t[n];t[n]=(...e)=>{let s=r.apply(t,e);return!1===s&&(s=i.apply(t,e)),s}}s.tokenizer=t}if(e.hooks){let t=this.defaults.hooks||new _t;for(let s in e.hooks){if(!(s in t))throw new Error(`hook '${s}' does not exist`);if(["options","block"].includes(s))continue;let n=s,r=e.hooks[n],i=t[n];_t.passThroughHooks.has(s)?t[n]=e=>{if(this.defaults.async&&_t.passThroughHooksRespectAsync.has(s))return(async()=>{let s=await r.call(t,e);return i.call(t,s)})();let n=r.call(t,e);return i.call(t,n)}:t[n]=(...e)=>{if(this.defaults.async)return(async()=>{let s=await r.apply(t,e);return!1===s&&(s=await i.apply(t,e)),s})();let s=r.apply(t,e);return!1===s&&(s=i.apply(t,e)),s}}s.hooks=t}if(e.walkTokens){let t=this.defaults.walkTokens,n=e.walkTokens;s.walkTokens=function(e){let s=[];return s.push(n.call(this,e)),t&&(s=s.concat(t.call(this,e))),s}}this.defaults={...this.defaults,...s}}),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return xt.lex(e,t??this.defaults)}parser(e,t){return $t.parse(e,t??this.defaults)}parseMarkdown(e){return(t,s)=>{let n={...s},r={...this.defaults,...n},i=this.onError(!!r.silent,!!r.async);if(!0===this.defaults.async&&!1===n.async)return i(new Error("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise."));if(typeof t>"u"||null===t)return i(new Error("marked(): input parameter is undefined or null"));if("string"!=typeof t)return i(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(t)+", string expected"));if(r.hooks&&(r.hooks.options=r,r.hooks.block=e),r.async)return(async()=>{let s=r.hooks?await r.hooks.preprocess(t):t,n=await(r.hooks?await r.hooks.provideLexer():e?xt.lex:xt.lexInline)(s,r),i=r.hooks?await r.hooks.processAllTokens(n):n;r.walkTokens&&await Promise.all(this.walkTokens(i,r.walkTokens));let o=await(r.hooks?await r.hooks.provideParser():e?$t.parse:$t.parseInline)(i,r);return r.hooks?await r.hooks.postprocess(o):o})().catch(i);try{r.hooks&&(t=r.hooks.preprocess(t));let s=(r.hooks?r.hooks.provideLexer():e?xt.lex:xt.lexInline)(t,r);r.hooks&&(s=r.hooks.processAllTokens(s)),r.walkTokens&&this.walkTokens(s,r.walkTokens);let n=(r.hooks?r.hooks.provideParser():e?$t.parse:$t.parseInline)(s,r);return r.hooks&&(n=r.hooks.postprocess(n)),n}catch(e){return i(e)}}}onError(e,t){return s=>{if(s.message+="\nPlease report this to https://github.com/markedjs/marked.",e){let e="<p>An error occurred:</p><pre>"+ut(s.message+"",!0)+"</pre>";return t?Promise.resolve(e):e}if(t)return Promise.reject(s);throw s}}};function vt(e,t){return yt.parse(e,t)}vt.options=vt.setOptions=function(e){return yt.setOptions(e),vt.defaults=yt.defaults,de(vt.defaults),vt},vt.getDefaults=pe,vt.defaults=ue,vt.use=function(...e){return yt.use(...e),vt.defaults=yt.defaults,de(vt.defaults),vt},vt.walkTokens=function(e,t){return yt.walkTokens(e,t)},vt.parseInline=yt.parseInline,vt.Parser=$t,vt.parser=$t.parse,vt.Renderer=bt,vt.TextRenderer=wt,vt.Lexer=xt,vt.lexer=xt.lex,vt.Tokenizer=mt,vt.Hooks=_t,vt.parse=vt,vt.options,vt.setOptions,vt.use,vt.walkTokens,vt.parseInline,$t.parse,xt.lex;
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const At=2;class St{constructor(e){}get _$AU(){return this._$AM._$AU}_$AT(e,t,s){this._$Ct=e,this._$AM=t,this._$Ci=s}_$AS(e,t){return this.update(e,t)}update(e,t){return this.render(...t)}}
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */class Tt extends St{constructor(e){if(super(e),this.it=j,e.type!==At)throw Error(this.constructor.directiveName+"() can only be used in child bindings")}render(e){if(e===j||null==e)return this._t=void 0,this.it=e;if(e===D)return e;if("string"!=typeof e)throw Error(this.constructor.directiveName+"() called with a non-string value");if(e===this.it)return this._t;this.it=e;const t=[e];return t.raw=t,this._t={_$litType$:this.constructor.resultType,strings:t,values:[]}}}Tt.directiveName="unsafeHTML",Tt.resultType=1;const Rt=(e=>(...t)=>({_$litDirective$:e,values:t}))(Tt);let Et=class extends ie{constructor(){super(...arguments),this._conversation=[],this._inputText="",this._isLoading=!1}static getStubConfig(){return{title:"Assist",show_tools:!0}}setConfig(e){if(!e)throw new Error("Invalid configuration");this._config=e,0===this._conversation.length&&(this._conversation=[{who:"hass",text:"How can I help you?",timestamp:new Date}])}firstUpdated(e){super.firstUpdated(e),this._scrollToBottom()}updated(e){super.updated(e),e.has("_conversation")&&this._scrollToBottom()}_scrollToBottom(){setTimeout(()=>{const e=this.shadowRoot?.querySelector(".conversation-container");e&&(e.scrollTop=e.scrollHeight)},100)}async _sendMessage(){if(!this._inputText.trim()||this._isLoading)return;const e={who:"user",text:this._inputText.trim(),timestamp:new Date};this._conversation=[...this._conversation,e];const t=this._inputText;this._inputText="",this._isLoading=!0;try{const e=await this._callAssistAPI(t),s={who:"hass",text:e.response.speech.plain.speech,timestamp:new Date,tool_calls:e.response.data?.tool_calls};this._conversation=[...this._conversation,s]}catch(e){console.error("Error calling Assist API:",e);const t={who:"hass",text:`Error: ${e instanceof Error?e.message:"Unknown error occurred"}`,timestamp:new Date,error:!0};this._conversation=[...this._conversation,t]}finally{this._isLoading=!1}}async _callAssistAPI(e){const t={type:"conversation/process",text:e};this._conversationId&&(t.conversation_id=this._conversationId),this._config?.pipeline_id&&(t.pipeline_id=this._config.pipeline_id);const s=await this.hass.callWS(t);return s.conversation_id&&(this._conversationId=s.conversation_id),s}_handleKeyPress(e){"Enter"!==e.key||e.shiftKey||(e.preventDefault(),this._sendMessage())}_toggleToolCall(e,t){const s=this._conversation[e];if(s.tool_calls&&s.tool_calls[t]){const n=[...this._conversation];n[e]={...s,tool_calls:s.tool_calls.map((e,s)=>s===t?{...e,expanded:!e.expanded}:e)},this._conversation=n}}_renderMarkdown(e){try{return vt.parse(e,{async:!1})}catch(t){return console.error("Error rendering markdown:",t),e}}_renderToolCalls(e,t){return this._config?.show_tools&&e.tool_calls&&0!==e.tool_calls.length?q`
      <div class="tool-calls">
        ${e.tool_calls.map((e,s)=>q`
            <div class="tool-call">
              <div
                class="tool-call-header"
                @click=${()=>this._toggleToolCall(t,s)}
              >
                <span class="tool-icon">${e.expanded?"▼":"▶"}</span>
                <span class="tool-name">Tool: ${e.tool_name}</span>
              </div>
              ${e.expanded?q`
                    <div class="tool-call-details">
                      <div class="tool-section">
                        <div class="tool-section-title">Input:</div>
                        <pre>${JSON.stringify(e.tool_input,null,2)}</pre>
                      </div>
                      ${e.tool_output?q`
                            <div class="tool-section">
                              <div class="tool-section-title">Output:</div>
                              <pre>${JSON.stringify(e.tool_output,null,2)}</pre>
                            </div>
                          `:""}
                    </div>
                  `:""}
            </div>
          `)}
      </div>
    `:q``}_renderMessage(e,t){const s="user"===e.who,n=`message ${s?"user-message":"assistant-message"} ${e.error?"error-message":""}`;return q`
      <div class=${n}>
        <div class="message-content">
          <div class="message-text">
            ${s?q`<div>${e.text}</div>`:q`<div class="markdown-content">
                  ${Rt(this._renderMarkdown(e.text))}
                </div>`}
          </div>
          ${this._renderToolCalls(e,t)}
        </div>
        <div class="message-timestamp">${e.timestamp.toLocaleTimeString()}</div>
      </div>
    `}render(){return this._config?q`
      <ha-card .header=${this._config.title}>
        <div class="card-content">
          <div class="conversation-container">
            ${this._conversation.map((e,t)=>this._renderMessage(e,t))}
            ${this._isLoading?q`
                  <div class="message assistant-message">
                    <div class="message-content">
                      <div class="loading-indicator">
                        <span class="dot"></span>
                        <span class="dot"></span>
                        <span class="dot"></span>
                      </div>
                    </div>
                  </div>
                `:""}
          </div>
          <div class="input-container">
            <textarea
              class="message-input"
              .value=${this._inputText}
              @input=${e=>this._inputText=e.target.value}
              @keypress=${this._handleKeyPress}
              placeholder=${this._config.placeholder||"Ask me anything..."}
              ?disabled=${this._isLoading}
              rows="1"
            ></textarea>
            <button
              class="send-button"
              @click=${this._sendMessage}
              ?disabled=${this._isLoading||!this._inputText.trim()}
            >
              <svg viewBox="0 0 24 24" width="24" height="24">
                <path fill="currentColor" d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
              </svg>
            </button>
          </div>
        </div>
      </ha-card>
    `:q``}};Et.styles=((e,...t)=>{const s=1===e.length?e[0]:t.reduce((t,s,n)=>t+(e=>{if(!0===e._$cssResult$)return e.cssText;if("number"==typeof e)return e;throw Error("Value passed to 'css' function must be a 'css' function result: "+e+". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.")})(s)+e[n+1],e[0]);return new i(s,e,n)})`
    :host {
      display: block;
    }

    ha-card {
      height: 100%;
      display: flex;
      flex-direction: column;
    }

    .card-content {
      display: flex;
      flex-direction: column;
      height: 600px;
      padding: 0;
    }

    .conversation-container {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: var(--card-background-color, #fff);
    }

    .message {
      display: flex;
      flex-direction: column;
      max-width: 80%;
      animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .user-message {
      align-self: flex-end;
    }

    .assistant-message {
      align-self: flex-start;
    }

    .message-content {
      padding: 12px 16px;
      border-radius: 16px;
      word-wrap: break-word;
    }

    .user-message .message-content {
      background: var(--primary-color, #03a9f4);
      color: var(--text-primary-color, #fff);
    }

    .assistant-message .message-content {
      background: var(--secondary-background-color, #f1f1f1);
      color: var(--primary-text-color, #000);
    }

    .error-message .message-content {
      background: var(--error-color, #f44336);
      color: #fff;
    }

    .message-timestamp {
      font-size: 0.75rem;
      color: var(--secondary-text-color, #888);
      margin-top: 4px;
      padding: 0 8px;
    }

    .user-message .message-timestamp {
      text-align: right;
    }

    .markdown-content {
      line-height: 1.6;
    }

    .markdown-content p {
      margin: 0.5em 0;
    }

    .markdown-content p:first-child {
      margin-top: 0;
    }

    .markdown-content p:last-child {
      margin-bottom: 0;
    }

    .markdown-content code {
      background: rgba(0, 0, 0, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 0.9em;
    }

    .markdown-content pre {
      background: rgba(0, 0, 0, 0.1);
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 8px 0;
    }

    .markdown-content pre code {
      background: none;
      padding: 0;
    }

    .markdown-content ul,
    .markdown-content ol {
      margin: 0.5em 0;
      padding-left: 1.5em;
    }

    .markdown-content h1,
    .markdown-content h2,
    .markdown-content h3,
    .markdown-content h4,
    .markdown-content h5,
    .markdown-content h6 {
      margin: 0.8em 0 0.4em 0;
      line-height: 1.3;
    }

    .markdown-content blockquote {
      border-left: 4px solid rgba(0, 0, 0, 0.2);
      margin: 0.5em 0;
      padding-left: 1em;
      color: var(--secondary-text-color, #666);
    }

    .markdown-content table {
      border-collapse: collapse;
      width: 100%;
      margin: 0.5em 0;
    }

    .markdown-content th,
    .markdown-content td {
      border: 1px solid rgba(0, 0, 0, 0.2);
      padding: 8px;
      text-align: left;
    }

    .markdown-content th {
      background: rgba(0, 0, 0, 0.1);
      font-weight: bold;
    }

    .tool-calls {
      margin-top: 12px;
      border-top: 1px solid rgba(0, 0, 0, 0.1);
      padding-top: 8px;
    }

    .tool-call {
      margin-bottom: 8px;
      border: 1px solid rgba(0, 0, 0, 0.15);
      border-radius: 8px;
      overflow: hidden;
    }

    .tool-call-header {
      padding: 8px 12px;
      background: rgba(0, 0, 0, 0.05);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      user-select: none;
      transition: background 0.2s;
    }

    .tool-call-header:hover {
      background: rgba(0, 0, 0, 0.1);
    }

    .tool-icon {
      font-size: 0.8em;
      width: 16px;
      display: inline-block;
    }

    .tool-name {
      font-weight: 500;
      font-size: 0.9em;
    }

    .tool-call-details {
      padding: 12px;
      background: rgba(0, 0, 0, 0.02);
    }

    .tool-section {
      margin-bottom: 8px;
    }

    .tool-section:last-child {
      margin-bottom: 0;
    }

    .tool-section-title {
      font-weight: 600;
      font-size: 0.85em;
      margin-bottom: 4px;
      color: var(--secondary-text-color, #666);
    }

    .tool-section pre {
      background: rgba(0, 0, 0, 0.1);
      padding: 8px;
      border-radius: 4px;
      overflow-x: auto;
      font-size: 0.85em;
      margin: 0;
      font-family: 'Courier New', monospace;
    }

    .input-container {
      display: flex;
      gap: 8px;
      padding: 16px;
      border-top: 1px solid var(--divider-color, #e0e0e0);
      background: var(--card-background-color, #fff);
    }

    .message-input {
      flex: 1;
      padding: 12px;
      border: 1px solid var(--divider-color, #e0e0e0);
      border-radius: 20px;
      font-family: inherit;
      font-size: 14px;
      resize: none;
      outline: none;
      min-height: 44px;
      max-height: 120px;
    }

    .message-input:focus {
      border-color: var(--primary-color, #03a9f4);
    }

    .message-input:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .send-button {
      width: 44px;
      height: 44px;
      border: none;
      border-radius: 50%;
      background: var(--primary-color, #03a9f4);
      color: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      flex-shrink: 0;
    }

    .send-button:hover:not(:disabled) {
      transform: scale(1.05);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .send-button:active:not(:disabled) {
      transform: scale(0.95);
    }

    .send-button:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .loading-indicator {
      display: flex;
      gap: 6px;
      padding: 8px;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--primary-text-color, #000);
      opacity: 0.4;
      animation: pulse 1.4s infinite ease-in-out;
    }

    .dot:nth-child(1) {
      animation-delay: -0.32s;
    }

    .dot:nth-child(2) {
      animation-delay: -0.16s;
    }

    @keyframes pulse {
      0%,
      80%,
      100% {
        opacity: 0.4;
        transform: scale(1);
      }
      40% {
        opacity: 1;
        transform: scale(1.2);
      }
    }
  `,e([ce({attribute:!1})],Et.prototype,"hass",void 0),e([he()],Et.prototype,"_config",void 0),e([he()],Et.prototype,"_conversation",void 0),e([he()],Et.prototype,"_conversationId",void 0),e([he()],Et.prototype,"_inputText",void 0),e([he()],Et.prototype,"_isLoading",void 0),Et=e([(e=>(t,s)=>{void 0!==s?s.addInitializer(()=>{customElements.define(e,t)}):customElements.define(e,t)})("homeassistant-assist-card")],Et),window.customCards=window.customCards||[],window.customCards.push({type:"homeassistant-assist-card",name:"Assist Card",description:"Custom Assist card with Markdown and Tool Call support"});export{Et as HomeAssistantAssistCard};
//# sourceMappingURL=homeassistant-assist-card.js.map
